"""HTML 报告生成：关键帧标注、标注视频、自包含报告（图片 base64 内嵌）。"""
import base64
import io
import os

import cv2
import numpy as np

from pose_extractor import IDX

PHASE_CN = {
    'setup': '预备 Setup',
    'ext': '发力伸展 Extension',
    'catch': '接铃 Catch',
    'stand': '站起 Stand',
}
PHASE_COLOR = {
    'setup': (120, 200, 120),
    'ext': (80, 160, 240),
    'catch': (60, 120, 255),
    'stand': (120, 220, 120),
}


def _img_b64(path):
    with open(path, 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


def _draw_skeleton(frame, lm, color=(0, 220, 0), thick=2):
    h, w = frame.shape[:2]
    def pt(i):
        return int(lm[i][0] * w), int(lm[i][1] * h)
    bones = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
             (25, 27), (26, 28), (27, 31), (28, 32), (27, 28)]
    for a, b in bones:
        cv2.line(frame, pt(a), pt(b), color, thick)
    for i in range(33):
        x, y = pt(i)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def _phase_frame(frames, lms, i, label, color):
    fr = frames[i].copy()
    if lms[i] is not None:
        _draw_skeleton(fr, lms[i])
    cv2.putText(fr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return fr


def _annotate_video(frames, lms, phases, fps, out_path):
    h, w = frames[0].shape[:2]
    # 先写临时 FMP4，再用 ffmpeg 转成浏览器可播的 H.264
    import subprocess
    tmp = out_path + '.raw.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
    phase_i = {'setup': int(phases[0] * fps), 'ext': int(phases[1] * fps),
               'catch': int(phases[2] * fps), 'stand': int(phases[3] * fps)}
    cur = 'setup'
    for i, fr in enumerate(frames):
        if i >= phase_i['ext'] and cur == 'setup':
            cur = 'ext'
        if i >= phase_i['catch'] and cur == 'ext':
            cur = 'catch'
        if i >= phase_i['stand'] and cur == 'catch':
            cur = 'stand'
        out = fr.copy()
        if lms[i] is not None:
            _draw_skeleton(out, lms[i], PHASE_COLOR[cur])
        cv2.putText(out, f'{PHASE_CN[cur]}  t={i/fps:.1f}s', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, PHASE_COLOR[cur], 2)
        vw.write(out)
    vw.release()
    # H.264 转码（浏览器兼容）；失败时保留原文件作为降级
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp, '-c:v', 'libx264', '-preset', 'veryfast',
             '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
             out_path],
            check=True, capture_output=True, timeout=300)
        os.remove(tmp)
    except Exception:
        os.replace(tmp, out_path)


def build_report(result, frames, lms, phases, fps, out, tag, annotate=True):
    jpgs = []
    for key, idx in [('setup', phases[0]), ('ext', phases[1]),
                     ('catch', phases[2]), ('stand', phases[3])]:
        i = min(len(frames) - 1, int(idx * fps))
        fr = _phase_frame(frames, lms, i, PHASE_CN[key], PHASE_COLOR[key])
        p = os.path.join(out, f'phase_{key}.jpg')
        cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpgs.append(p)

    mp4 = None
    if annotate:
        mp4 = os.path.join(out, f'{tag}_annotated.mp4')
        _annotate_video(frames, lms, phases, fps, mp4)

    comps = result['components']
    comp_cn = {'extension': '发力充分性', 'bar_path': '杠铃贴身', 'catch': '接铃锁定',
               'squat': '下蹲深度', 'trunk': '躯干稳定', 'flow': '动作连贯',
               'double_pull': '二次拉检测', 'timing': '发力时机',
               'foot': '脚掌稳定', 'sequence': '伸展顺序'}
    rows = ''.join(
        f'<tr><td>{cn}</td><td>'
        + (f'<span class="score">{comps[k]["score"]}</span>'
           if comps[k].get('reliable', True)
           else f'<span style="color:#cf222e;">样本受限</span>（原始 {comps[k]["score"]}）')
        + f'</td><td>{comps[k]["metric"]}</td>'
        f'<td>{comps[k]["suggestion"] or "—"}</td></tr>'
        for k, cn in comp_cn.items() if k in comps
    )
    imgs = ''.join(
        f'<div style="flex:1;min-width:180px;"><img src="{_img_b64(p)}" '
        f'style="width:100%;border-radius:8px;"><div style="font-size:12px;'
        f'color:#57606a;margin-top:4px;">{PHASE_CN[k]}</div></div>'
        for k, p in zip(['setup', 'ext', 'catch', 'stand'], jpgs)
    )
    notes = ''.join(f'<li>{n}</li>' for n in result['notes'])
    if mp4 and os.path.exists(mp4):
        # 视频以 base64 内嵌，报告自包含、双击即可播放；
        # poster 用接铃关键帧，未播放时也有画面；muted 自动播放兜底
        import base64
        b64 = base64.b64encode(open(mp4, 'rb').read()).decode()
        poster = _img_b64(jpgs[2])  # phase_catch
        mp4_html = ('<video controls muted autoplay playsinline loop '
                    f'poster="{poster}" preload="metadata" '
                    'style="width:100%;max-width:720px;background:#000;">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')
    else:
        mp4_html = '<p>未生成标注视频</p>'

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{tag} 举重打分报告</title>
<style>
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;background:#f6f8fa;color:#1f2328;}}
.wrap{{max-width:900px;margin:0 auto;padding:24px 16px;}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:20px;margin-bottom:16px;}}
h1{{font-size:22px;margin:0 0 4px;}} h2{{font-size:16px;margin:0 0 12px;}}
.big{{font-size:40px;font-weight:800;}} .g{{color:#1a7f37;}}
table{{width:100%;border-collapse:collapse;font-size:14px;}}
th,td{{border-bottom:1px solid #eaeef2;padding:8px 6px;text-align:left;}}
th{{color:#57606a;font-weight:600;}}
.score{{font-weight:700;color:#0969da;}}
.imgs{{display:flex;gap:10px;flex-wrap:wrap;}}
.bar{{height:8px;background:#eaeef2;border-radius:4px;position:relative;}}
.bar i{{position:absolute;left:0;top:0;height:8px;border-radius:4px;background:#2da44e;}}
</style></head><body><div class="wrap">
<div class="card">
  <h1>举重打分报告 · {result['lift']}</h1>
  <div style="color:#57606a;font-size:13px;">{result['video']} · 阶段(秒):
  setup={result['phases_sec']['setup']} ext={result['phases_sec']['ext']}
  catch={result['phases_sec']['catch']} stand={result['phases_sec']['stand']}</div>
  <div style="margin:16px 0;"><span class="big">{result['total_score']}</span>
  <span style="font-size:16px;color:#57606a;">/100</span>
  <span class="g" style="margin-left:10px;font-size:18px;">{result['grade']}</span></div>
  <div class="bar"><i style="width:{result['total_score']}%;"></i></div>
</div>
<div class="card"><h2>分项得分与改进建议</h2>
<table><tr><th>分项</th><th>得分</th><th>实测指标</th><th>改进建议</th></tr>{rows}</table></div>
<div class="card"><h2>关键帧</h2><div class="imgs">{imgs}</div></div>
<div class="card"><h2>标注视频</h2>{mp4_html}</div>
<div class="card"><h2>说明</h2><ul style="font-size:13px;color:#57606a;">{notes}</ul></div>
</div></body></html>"""
    html_path = os.path.join(out, f'{tag}_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html_path, jpgs
