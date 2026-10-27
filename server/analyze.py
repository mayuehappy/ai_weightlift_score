"""举重 AI 打分原型 · 主入口。

用法:
    python3 analyze.py <video> [--lift snatch|clean] [--start S] [--end E]
                          [--out <目录>] [--no-annotate] [--model <task文件>]

产出（--out 目录，默认 output/）:
    <视频名>_<lift>_result.json   结构化打分结果
    <视频名>_<lift>_report.html   自包含 HTML 报告
    <视频名>_<lift>_annotated.mp4 骨架/关节角标注视频
    phase_setup/ext/catch/stand.jpg 各阶段关键帧
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from pose_extractor import PoseExtractor
from scoring import (compute_features, pick_load_side, finalize_features,
                     ema_features, detect_phases, score_snatch, score_clean,
                     _tech_scores, compute_score, build_suggestions,
                     DEFAULT_WEIGHTS)
from report import build_report


def analyze(video, lift='snatch', start=None, end=None, out='output',
            annotate=True, model='pose_landmarker_full.task'):
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    tag = f'{stem}_{lift}'

    print(f'[1/5] 姿态提取 {video} ...', flush=True)
    t0 = time.time()
    extractor = PoseExtractor(model_path=model)
    frames, lms, fps, total = extractor.extract(video, start=start, end=end)
    print(f'      帧数 {len(frames)}，fps={fps:.2f}，耗时 {time.time()-t0:.0f}s', flush=True)

    feats = []
    for lm in lms:
        if lm is None:
            feats.append(None)
            continue
        f = compute_features(lm)
        feats.append(f if f and f['vis'] > 0.4 else None)

    # 取连续有效段
    valid = [i for i, f in enumerate(feats) if f]
    if len(valid) < 8:
        raise ValueError('有效姿态帧太少（<8），请检查视频构图（需全身入镜、侧面视角）')

    side = pick_load_side([feats[i] for i in valid])
    print(f'[2/5] 持铃侧判定: {"左" if side=="l" else "右"}', flush=True)
    fv = [feats[i] for i in valid]
    # 站立腿长：取腿长 P95（站立时髋-踝距离最大），用作杠铃高度/距离的稳定标尺
    stand_leg = float(np.percentile([f['leg'] for f in fv], 95))
    fv = finalize_features(fv, side, stand_leg=stand_leg)

    print('[3/5] 阶段切分 ...', flush=True)
    # 阶段切分用插值后但未平滑的特征（detect_phases 内部自带窗口平滑），
    # 避免 EMA 对序列开头的偏置把发力峰值定位带偏
    bar_h = [f['bar_h'] for f in fv]
    hip = [f['hip'] for f in fv]
    knee = [f['knee'] for f in fv]
    trunk = [f['trunk'] for f in fv]
    phases = detect_phases(bar_h, hip, knee, trunk, fps)
    if phases is None:
        raise ValueError('未识别出完整的"低位→高位"试举结构。窗口需只含一次完整动作（杠铃从低位到高位）')
    setup_s, ext_s, catch_s, stand_s = phases
    print(f'      阶段(秒): setup={setup_s:.1f} ext={ext_s:.1f} '
          f'catch={catch_s:.1f} stand={stand_s:.1f}', flush=True)

    # 特征级 EMA 平滑（仅用于六维打分，稳定数值类指标）：
    # 时序判定（技术细节四维）仍用原始特征，避免滤波器的相位滞后偏置峰值位置
    fv_raw = fv
    fv_sm = ema_features(fv)
    fv = fv_sm

    # 对齐到原视频时间轴（截取窗口偏移）
    t_off = 0.0 if start is None else start

    print('[4/5] 规则打分 ...', flush=True)
    scorer = score_snatch if lift == 'snatch' else score_clean
    comps = scorer(fv_sm, phases, fps)
    # 技术细节四维用原始特征（时序判定不受 EMA 相位滞后影响）
    comps.update(_tech_scores(fv_raw, phases, fps))
    if comps is None:
        raise ValueError('阶段窗口不足以计算分项。请确保窗口内包含一次完整的试举动作')

    # 可靠性标记：杠铃贴身依赖"过膝段"样本；发力过快/背后机位时腕点跟丢，
    # 过膝段与接铃窗口出现物理上不可能的贴身距离 → 对应项自动排除出总分
    pull_frames = [f for f in fv[int(phases[0] * fps): int(phases[1] * fps) + int(0.5 * fps)]
                   if 0.35 <= f['bar_h'] <= 0.75]
    n_pull = len(pull_frames)
    pull_med = float(np.median([f['bar_gap'] for f in pull_frames])) if pull_frames else 99.0
    bar_path_limited = n_pull < 3 or pull_med > 0.6
    if bar_path_limited:
        print(f'      提示：过膝段有效样本 {n_pull} 帧、贴身距离中位 {pull_med:.2f}，'
              f'杠铃贴身项样本不足，自动排除出总分', flush=True)

    catch_frames = fv[int(phases[2] * fps): int(phases[3] * fps) + int(0.6 * fps)]
    catch_gap = float(np.median([f['bar_gap'] for f in catch_frames])) if catch_frames else 99.0
    catch_limited = catch_gap > 0.6
    if catch_limited:
        print(f'      提示：接铃窗口贴身距离中位 {catch_gap:.2f}，'
              f'接铃锁定项不可信，自动排除出总分', flush=True)

    # 技术细节维度：样本不足时未产生分数，自动不参与总分（compute_score 自动跳过）
    tech_keys = ('double_pull', 'timing', 'foot', 'sequence')
    tech_missing = [k for k in tech_keys if k not in comps]
    if tech_missing:
        print(f'      提示：技术细节维度 {tech_missing} 样本不足，未参与评分', flush=True)

    # 站姿稳定：踝点水平漂移过大（相机晃动/大幅挪步）时，侧机位 2D 无法分离原因 → 剔除
    foot_std = comps.get('_foot_std', 0.0)
    foot_limited = bool(foot_std) and foot_std > 0.15
    if foot_limited:
        print(f'      提示：踝点水平漂移 std={foot_std:.3f}（腿长），站姿稳定项不可信，自动排除出总分', flush=True)

    unreliable = []
    if bar_path_limited:
        unreliable.append('bar_path')
    if catch_limited:
        unreliable.append('catch')
    if foot_limited:
        unreliable.append('foot')
    total_score, grade = compute_score(comps)
    if unreliable:
        # 剔除不可靠分项并按剩余权重重新归一
        w = {k: v for k, v in DEFAULT_WEIGHTS.items() if k not in unreliable}
        total_score, grade = compute_score(comps, w)
        print(f'      剔除后总分 {total_score}（{grade}）', flush=True)
    metrics = {
        'extension': f"{comps['_ext_hip']:.0f}°",
        'bar_path': f"{comps['_bar_gap']:.2f}",
        'squat': f"{comps['_catch_knee']:.0f}°",
        'trunk': f"{comps['_trunk_max']:.0f}°",
        'catch': '见锁定分',
        'flow': f"{comps['flow']:.0f}",
        'double_pull': f"{comps.get('_v_min', float('nan')):.3f}",
        'timing': f"{comps.get('_dt', float('nan')):.2f}",
        'foot': f"{comps.get('_foot_std', float('nan')):.3f}",
    }
    if 'sequence' in comps:
        dt = comps['_dt_hipknee']
        if dt <= 0.08:
            metrics['sequence'] = '髋先于膝'
        elif dt > 0.4:
            metrics['sequence'] = f'膝先于髋 {dt:.1f}s'
        else:
            metrics['sequence'] = f'髋晚于膝 {dt:.1f}s'
    # 建议文案里已有单位，传入裸数值避免重复单位（如 "32°°"）
    sug_metrics = {k: (v[:-1] if isinstance(v, str) and v.endswith('°') else v)
                   for k, v in metrics.items()}
    sug = build_suggestions(comps, sug_metrics)
    print(f'      总分 {total_score}（{grade}）', flush=True)

    occl_sug = '腕点被躯干遮挡/跟丢（背后机位或发力过快），该项暂不计入总分'
    unreli_note = []
    if bar_path_limited:
        unreli_note.append('杠铃贴身：过膝段有效样本不足（发力过快/腕点抖动），已排除出总分')
    if catch_limited:
        unreli_note.append('接铃锁定：接铃窗口腕点不可信（背后机位或遮挡），已排除出总分')
    if tech_missing:
        unreli_note.append('技术细节维度（' + '、'.join(tech_missing) + '）：样本不足未参与评分')
    if foot_limited:
        unreli_note.append('站姿稳定：踝点水平漂移明显（相机晃动或大幅挪步），侧机位无法分离原因，已排除出总分')
    if not unreli_note:
        unreli_note.append('全部有效分项正常参与评分。')

    comp_cn = {'extension': '发力充分性', 'bar_path': '杠铃贴身', 'catch': '接铃锁定',
               'squat': '下蹲深度', 'trunk': '躯干稳定', 'flow': '动作连贯',
               'double_pull': '二次拉检测', 'timing': '发力时机',
               'foot': '脚掌稳定', 'sequence': '伸展顺序'}
    components = {}
    for k in comp_cn:
        if k not in comps:
            continue
        if k == 'bar_path':
            components[k] = {'score': round(comps[k], 1), 'metric': metrics.get(k, ''),
                             'reliable': not bar_path_limited,
                             'suggestion': occl_sug if bar_path_limited else
                                           next((s for kk, _, s in sug if kk == k), None)}
        elif k == 'catch':
            components[k] = {'score': round(comps[k], 1), 'metric': metrics.get(k, ''),
                             'reliable': not catch_limited,
                             'suggestion': occl_sug if catch_limited else
                                           next((s for kk, _, s in sug if kk == k), None)}
        elif k == 'foot':
            components[k] = {'score': round(comps[k], 1), 'metric': metrics.get(k, ''),
                             'reliable': not foot_limited,
                             'suggestion': '踝点水平漂移明显（机位晃动或挪步），侧机位无法评估站姿稳定'
                                           if foot_limited else
                                           next((s for kk, _, s in sug if kk == k), None)}
        else:
            components[k] = {'score': round(comps[k], 1), 'metric': metrics.get(k, ''),
                             'reliable': True,
                             'suggestion': next((s for kk, _, s in sug if kk == k), None)}
    result = {
        'video': video, 'lift': lift, 'fps': fps,
        'window': {'start': start, 'end': end},
        'total_score': total_score, 'grade': grade,
        'view_quality': {'bar_path_limited': bar_path_limited,
                         'catch_limited': catch_limited,
                         'tech_missing': tech_missing,
                         'note': '；'.join(unreli_note)},
        'phases_sec': {'setup': round(setup_s + t_off, 1),
                       'ext': round(ext_s + t_off, 1),
                       'catch': round(catch_s + t_off, 1),
                       'stand': round(stand_s + t_off, 1)},
        'components': components,
        'notes': [
            '规则打分 MVP：MediaPipe 只负责姿态提取，打分与建议为人工规则，示意性质。',
            f'本视频按 {lift} 口径评分；翻铃的"接铃锁定"评估前架位而非过顶。',
            '；'.join(unreli_note) + '（被排除项分数仅供参考，建议侧面机位、慢动作或更高帧率复核。）'
            if unreli_note != ['全部有效分项正常参与评分。'] else '全部有效分项正常参与评分。',
            '技术细节四维（二次拉/发力时机/脚掌稳定/伸展顺序）为 v2 规则新增，参与总分平均。',
        ],
    }
    rp = os.path.join(out, f'{tag}_result.json')
    with open(rp, 'w', encoding='utf-8') as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    print('[5/5] 生成报告与标注 ...', flush=True)
    report_path, key_jpgs = build_report(result, frames, lms, phases, fps,
                                         out, tag, annotate=annotate)
    print('完成。产物:', flush=True)
    for p in [rp, report_path] + key_jpgs:
        print('  ', p, flush=True)
    return rp


def main():
    ap = argparse.ArgumentParser(description='举重 AI 打分原型')
    ap.add_argument('video')
    ap.add_argument('--lift', choices=['snatch', 'clean'], default='snatch')
    ap.add_argument('--start', type=float, default=None)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--out', default='output')
    ap.add_argument('--no-annotate', action='store_true')
    ap.add_argument('--model', default='pose_landmarker_full.task')
    args = ap.parse_args()
    analyze(args.video, args.lift, args.start, args.end, args.out,
            not args.no_annotate, args.model)


if __name__ == '__main__':
    main()
