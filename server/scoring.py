"""规则打分：特征计算、阶段切分、抓举/翻铃打分与建议生成。

约定：
- 角度单位：度。
- 躯干角：相对垂直线的倾角，0°=直立。
- 髋角/膝角：关节点内角，180°=伸直，越小越弯曲。
- bar_h：杠铃高度代理 = 腕点相对踝部的高度 / 腿长（髋-踝距离）。
  地面预备时约 0，举过头顶时 >1。全程归一化，跨视频可比。
- 所有分项 0-100，越高越好；总分按权重合成，映射为等级。
"""
import numpy as np

IDX_NOSE = 0
IDX_SHOULDER = (11, 12)
IDX_ELBOW = (13, 14)
IDX_WRIST = (15, 16)
IDX_HIP = (23, 24)
IDX_KNEE = (25, 26)
IDX_ANKLE = (27, 28)

DEFAULT_WEIGHTS = {
    'extension': 0.20,   # 发力充分性
    'bar_path': 0.20,    # 杠铃贴身
    'catch': 0.20,       # 接铃锁定（抓举=过顶锁定，翻铃=前架位）
    'squat': 0.15,       # 下蹲深度
    'trunk': 0.15,       # 躯干稳定
    'flow': 0.10,        # 动作连贯
    'double_pull': 0.05, # 二次拉检测（技术细节）
    'timing': 0.05,      # 发力时机
    'foot': 0.05,        # 脚掌稳定
    'sequence': 0.05,    # 伸展顺序
}


def ema_features(feats, alpha=0.45):
    """特征序列 EMA 平滑：消除单帧检测抖动，保留动作真实轨迹。

    对慢变指标（关节角、杠铃位置、踝位）逐通道做一阶 EMA。
    alpha 越大越跟手，越小越平滑；0.45 在降噪与保真间取平衡。
    """
    if not feats:
        return feats
    fields = ['hip', 'knee_l', 'knee_r', 'elbow_l', 'elbow_r', 'trunk',
              'bar_h', 'wrist_h', 'ank_l_x', 'ank_r_x',
              'hip_x', 'hip_y', 'ank_y', 'ank_x', 'knee', 'elbow',
              'bar_gap', 'bar_over_base']
    keys = fields + [k for k in feats[0] if k not in fields]
    out = []
    prev = {k: None for k in keys}
    for f in feats:
        nf = dict(f)
        for k in keys:
            v = f.get(k)
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if prev[k] is None:
                nf[k] = float(v)
            else:
                nf[k] = alpha * float(v) + (1.0 - alpha) * prev[k]
            prev[k] = nf[k]
        out.append(nf)
    return out


# ---------------------------------------------------------------- 基础几何
def _angle(a, b, c):
    """三点内角（度），b 为顶点。"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _trunk_angle_from_vertical(hip, shoulder):
    """躯干相对垂直线的倾角：0=直立，前倾为正。"""
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    if w < 2:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def _err_score(val, target, tol, max_err, min_score=0.0):
    """目标值附近容差内满分，超出按线性惩罚到 min_score。"""
    if np.isnan(val):
        return min_score
    d = abs(val - target)
    if d <= tol:
        return 100.0
    if max_err <= tol:
        return 100.0 if d <= tol else min_score
    return max(min_score, 100.0 * max(0.0, 1.0 - (d - tol) / (max_err - tol)))


# ---------------------------------------------------------------- 特征计算
def compute_features(lm):
    """lm: [(x,y,z,visibility)*33]，归一化坐标。返回特征字典或 None。"""
    def p(i):
        return (lm[i][0], lm[i][1], lm[i][2], lm[i][3])

    hip = [(p(23)[0] + p(24)[0]) / 2, (p(23)[1] + p(24)[1]) / 2]
    sh = [(p(11)[0] + p(12)[0]) / 2, (p(11)[1] + p(12)[1]) / 2]
    ank = [(p(27)[0] + p(28)[0]) / 2, (p(27)[1] + p(28)[1]) / 2]
    # 归一化坐标 y 向下：踝在髋下方，腿长 = 踝_y - 髋_y
    leg = max(ank[1] - hip[1], 1e-3)

    feat = {
        'hip': _angle(sh[:2], hip, p(25)[:2]),       # 肩-髋-膝（矢状面髋角）
        'knee_l': _angle(p(23)[:2], p(25)[:2], p(27)[:2]),
        'knee_r': _angle(p(24)[:2], p(26)[:2], p(28)[:2]),
        'elbow_l': _angle(p(11)[:2], p(13)[:2], p(15)[:2]),
        'elbow_r': _angle(p(12)[:2], p(14)[:2], p(16)[:2]),
        'ankle_l': _angle(p(25)[:2], p(27)[:2], p(31)[:2]),   # 膝-踝-脚
        'ankle_r': _angle(p(26)[:2], p(28)[:2], p(32)[:2]),
        'trunk': _trunk_angle_from_vertical(hip, sh),
        # 原始高度量：腕相对踝的竖直距离、当前腿长（后续用序列级站立腿长归一化）
        'wrist_h': (ank[1] - (p(16)[1] + p(15)[1]) / 2),
        'leg': max(ank[1] - hip[1], 1e-3),
        'bar_h': (ank[1] - (p(16)[1] + p(15)[1]) / 2) / max(ank[1] - hip[1], 1e-3),
        'w_l_y': p(15)[1], 'w_r_y': p(16)[1],
        'w_l_x': p(15)[0], 'w_r_x': p(16)[0],
        'ank_l_x': p(27)[0], 'ank_r_x': p(28)[0],
        'hip_y': hip[1], 'hip_x': hip[0], 'ank_y': ank[1],
        'vis': sum(q[3] for q in lm) / len(lm),
    }
    return feat


def pick_load_side(feats):
    """单臂动作（壶铃/哑铃）：按腕点竖直行程更大的一侧判定持铃臂。"""
    rl = max(f['w_l_y'] for f in feats) - min(f['w_l_y'] for f in feats)
    rr = max(f['w_r_y'] for f in feats) - min(f['w_r_y'] for f in feats)
    return 'l' if rl >= rr else 'r'


def finalize_features(feats, side='r', stand_leg=None):
    """补充单侧/双侧统一的特征：膝角、肘角、杠铃贴身与支撑面偏移。
    stand_leg: 序列级站立腿长（P95），缺省用逐帧腿长。
    """
    for f in feats:
        f['knee'] = f['knee_l'] if side == 'l' else f['knee_r']
        f['elbow'] = f['elbow_l'] if side == 'l' else f['elbow_r']
        f['ankle'] = f['ankle_l'] if side == 'l' else f['ankle_r']
        f['ank_x'] = f['ank_l_x'] if side == 'l' else f['ank_r_x']
        # 杠铃高度统一用站立腿长归一化，避免深蹲时腿长压缩导致数值爆炸
        leg = stand_leg if stand_leg else max(f['ank_y'] - f['hip_y'], 1e-3)
        f['bar_h'] = f['wrist_h'] / leg
        # 杠铃相对躯干水平距离：腕点 x 与髋中点 x 之差（归一化到腿长）
        wx = f['w_l_x'] if side == 'l' else f['w_r_x']
        f['bar_gap'] = abs(wx - f['hip_x']) / leg
        # 杠铃相对支撑面（踝）的水平偏移：过顶时应在踝正上方
        f['bar_over_base'] = abs(wx - f['ank_x']) / leg
    return feats


# ---------------------------------------------------------------- 阶段切分
def detect_phases(bar_h, hip, knee, trunk, fps, catch_win=2.5, stand_win=3.5,
                  pull_win=3.0, max_barh=2.5):
    """把一次试举切成 setup/ext/catch/stand 四个阶段边界（秒）。
    返回 (setup_s, ext_s, catch_s, stand_s)；找不到则返回 None。
    """
    n = len(bar_h)
    if n < 8:
        return None
    bh = smooth(np.clip(bar_h, 0.0, max_barh), 5)
    hp = smooth(hip, 5)

    # 1) 一次完整"低位→高位"：bar_h 全局最小，拉铃窗口内找峰值（限制在 pull_win 内，
    #    避免片尾跟踪残差被误当顶峰）
    i_min = int(np.argmin(bh))
    i_max_ub = min(n, i_min + int(pull_win * fps))
    if i_max_ub - i_min < 3:
        return None
    i_max = int(np.argmax(bh[i_min:i_max_ub])) + i_min

    # 2) setup：杠铃离开底部前 = 窗口起点附近（取 i_min 前推，裁剪到窗口起点）
    setup_s = max(0.0, (i_min - 2) / fps)

    # 3) ext：发力伸展 = 在 bar 上升区间内髋角达到峰值
    seg = bh[i_min:i_max + 1]
    hp_seg = hp[i_min:i_max + 1]
    if len(hp_seg) < 3:
        return None
    i_ext = i_min + int(np.argmax(hp_seg))
    ext_s = i_ext / fps

    # 4) catch：接铃 = 杠铃到顶后第一次深蹲（膝角最小），在 catch_win 秒窗口内找
    kn = smooth(knee, 5) if len(knee) == n else np.asarray(knee)
    i_catch = None
    for i in range(i_max, min(n, i_max + int(catch_win * fps))):
        if kn[i] < kn[i_catch] if (i_catch is not None) else True:
            # 需确实下蹲（膝角 < 135°）才视为接铃深蹲
            if kn[i] < 135 and (i_catch is None or kn[i] < kn[i_catch]):
                i_catch = i
    if i_catch is None:
        i_catch = i_max
    catch_s = i_catch / fps

    # 5) stand：站起 = 接铃后躯干回到直立（躯干角 < 15°），stand_win 秒内
    tr = smooth(trunk, 5) if len(trunk) == n else np.asarray(trunk)
    i_stand = None
    for i in range(i_catch, min(n, i_catch + int(stand_win * fps))):
        if tr[i] < 15.0:
            i_stand = i
            break
    if i_stand is None:
        i_stand = min(n - 1, i_catch + int(stand_win * fps))
    stand_s = i_stand / fps

    return (setup_s, ext_s, catch_s, stand_s)


# ---------------------------------------------------------------- 打分
def _slice(feats, t0, t1, fps, min_dur=0.6):
    """取 [t0, t1) 帧区间；区间过短时向后扩展保证至少 min_dur 秒。"""
    a = max(0, int(t0 * fps))
    b = max(a + 1, int(t1 * fps))
    if b - a < int(min_dur * fps):
        b = min(len(feats), a + int(min_dur * fps))
    return feats[a: b]


def _tech_scores(feats, phases, fps):
    """技术细节四维：二次拉 / 发力时机 / 脚掌稳定 / 伸展顺序。

    样本不足的维度不出现在返回 dict 中，由上层按"剔除项"处理。
    feats 为 EMA 平滑后的特征序列；阶段边界由未平滑特征切出（避免偏置）。
    """
    n = len(feats)
    setup, ext, catch, stand = phases
    bh = np.asarray([f['bar_h'] for f in feats])
    bh_s = smooth(np.clip(bh, 0.0, 2.5), 5)
    # 拉铃段 = [setup, catch]：杠铃从低位升到顶
    a = int(setup * fps)
    b = min(n, int(catch * fps))
    rise = bh_s[a: b + 1]
    out = {}

    # 1) 二次拉：杠铃上升段速度最低点（回落程度）
    #    v_min 越负 = 停顿/回落越明显；标准动作几乎不回落
    if len(rise) >= 6:
        v = np.diff(rise)
        v_min = float(np.min(v))
        out['double_pull'] = _err_score(v_min, 0.0, 0.02, 0.10)
        out['_v_min'] = v_min

    # 2) 发力时机：速度峰值时刻与发力时刻（髋角峰值）的时间差
    #    教科书节奏：杠铃最快时正是顶髋发力时（峰值应落在 ext ±0.2s 内）
    if len(rise) >= 6:
        v = np.diff(rise)
        i_vmax = int(np.argmax(v))
        t_vmax = (a + i_vmax) / fps
        dt = abs(t_vmax - ext)
        out['timing'] = _err_score(dt, 0.0, 0.2, 0.8)
        out['_dt'] = dt

    # 3) 站姿稳定：踝点绝对水平位置的波动（归一化到站立腿长）
    #    脚踩地不动 → 踝绝对 x 基本不变（姿态变化不影响）；波动大 = 挪步或机位晃动
    #    阈值 0.15 以上视为"漂移明显"，由上层剔除出总分（侧机位 2D 下无法分离原因）
    if n >= 8:
        ank = [f['ank_x'] for f in feats]
        leg = float(np.percentile([f['leg'] for f in feats], 95))
        std = float(np.std(ank)) / max(leg, 1e-3)
        out['foot'] = _err_score(std, 0.0, 0.04, 0.15)
        out['_foot_std'] = std

    # 4) 伸展顺序：发力时刻前后窗口内，髋先伸、膝后伸（近端到远端）
    #    用"达到 90% 峰值"的穿越时间而非 argmax：抗平顶/检测噪声
    #    （踝关节在侧机位 2D 下脚趾点不可靠，不参与本维度判定）
    t_lo = max(setup, ext - 0.4)
    t_hi = min(stand, ext + 0.4)
    ext_win = _slice(feats, t_lo, t_hi, fps)
    if len(ext_win) >= 4:
        def _t90(vals):
            m = max(vals)
            if m <= 0:
                return 0.0
            thr = 0.9 * m
            for i, v in enumerate(vals):
                if v >= thr:
                    return i / fps
            return len(vals) / fps
        t_hip = _t90([f['hip'] for f in ext_win])
        t_knee = _t90([f['knee'] for f in ext_win])
        dt = t_hip - t_knee  # 负 = 髋先于膝（正确）
        out['sequence'] = _err_score(dt, 0.0, 0.08, 0.4)
        out['_dt_hipknee'] = dt
    return out


def score_snatch(feats, phases, fps):
    """抓举 6 分项。phases=(setup,ext,catch,stand) 秒。"""
    setup, ext, catch, stand = phases
    ext_win = _slice(feats, ext, catch, fps)
    catch_win = _slice(feats, catch, stand, fps)
    setup_win = _slice(feats, setup, ext, fps)
    all_win = feats

    if not ext_win or not catch_win:
        return None

    ext_hip = max(f['hip'] for f in ext_win)
    s_ext = _err_score(ext_hip, 170, 10, 45)         # 发力充分性：髋伸展≥170°（完全伸展）
    # 杠铃贴身：只在"杠铃过膝/大腿段"（bar_h 0.35~0.75）评估，此时杠铃应贴腿
    pull = [f for f in feats[int(phases[0] * fps): int(phases[1] * fps) + int(0.5 * fps)]
            if 0.35 <= f['bar_h'] <= 0.75]
    bar_gap = (np.median([f['bar_gap'] for f in pull]) if pull
               else np.median([f['bar_gap'] for f in ext_win]))
    s_bar = _err_score(bar_gap, 0.15, 0.10, 0.6)     # 杠铃贴身：过膝段水平距离小
    catch_knee = min(f['knee'] for f in catch_win)
    s_squat = _err_score(catch_knee, 70, 20, 70)     # 下蹲深度：接铃深蹲膝角
    # 接铃锁定（抓举=过顶锁定）：肘伸直 + 杠铃足够高（过顶）+ 杠铃在支撑面正上方
    elbow = np.median([f['elbow'] for f in catch_win])
    bh_c = np.median([f['bar_h'] for f in catch_win])
    base_c = np.median([f['bar_over_base'] for f in catch_win])
    lock = (0.5 * _err_score(elbow, 175, 10, 45)
            + 0.3 * _err_score(bh_c, 1.2, 0.2, 0.6)
            + 0.2 * _err_score(base_c, 0.15, 0.10, 0.5))
    lift_win = _slice(feats, ext, stand + 1.0, fps)  # 躯干稳定只看发力→站起
    trunk_max = float(np.percentile([f['trunk'] for f in lift_win], 95))
    s_trunk = _err_score(trunk_max, 5, 10, 45)       # 躯干稳定：发力→站起不歪（P95 抗尖峰）
    # 动作连贯：杠铃路径平滑（相邻帧 bar_h 变化率的抖动）
    s_flow = _flow_score(feats)

    comps = {
        'extension': s_ext, 'bar_path': s_bar, 'catch': lock,
        'squat': s_squat, 'trunk': s_trunk, 'flow': s_flow,
        '_ext_hip': ext_hip, '_bar_gap': bar_gap, '_catch_knee': catch_knee,
        '_trunk_max': trunk_max, '_elbow': elbow, '_barh_catch': bh_c,
    }
    return comps


def score_clean(feats, phases, fps):
    """翻铃（clean）6 分项：接铃锁定改为"前架位"（肘抬高、杠铃贴身于肩）。"""
    setup, ext, catch, stand = phases
    ext_win = _slice(feats, ext, catch, fps)
    catch_win = _slice(feats, catch, stand, fps)
    lift_win = _slice(feats, ext, stand + 1.0, fps)   # 躯干稳定只看发力→站起
    all_win = feats

    if not ext_win or not catch_win:
        return None

    ext_hip = max(f['hip'] for f in ext_win)
    s_ext = _err_score(ext_hip, 170, 10, 45)
    # 杠铃贴身：过膝/大腿段评估（翻铃同理）
    pull = [f for f in feats[int(phases[0] * fps): int(phases[1] * fps) + int(0.5 * fps)]
            if 0.35 <= f['bar_h'] <= 0.75]
    bar_gap = (np.median([f['bar_gap'] for f in pull]) if pull
               else np.median([f['bar_gap'] for f in ext_win]))
    s_bar = _err_score(bar_gap, 0.15, 0.10, 0.6)
    catch_knee = min(f['knee'] for f in catch_win)
    s_squat = _err_score(catch_knee, 90, 25, 70)     # 翻铃可半蹲/全蹲，容差放宽
    # 前架位：接铃瞬间杠铃贴近躯干（bar_gap 小）且躯干基本直立
    rack_gap = np.median([f['bar_gap'] for f in catch_win])
    rack_trunk = np.median([f['trunk'] for f in catch_win])
    s_rack = 0.6 * _err_score(rack_gap, 0.15, 0.10, 0.5) + 0.4 * _err_score(rack_trunk, 5, 10, 45)
    trunk_max = float(np.percentile([f['trunk'] for f in lift_win], 95))
    s_trunk = _err_score(trunk_max, 5, 10, 45)
    s_flow = _flow_score(feats)

    comps = {
        'extension': s_ext, 'bar_path': s_bar, 'catch': s_rack,
        'squat': s_squat, 'trunk': s_trunk, 'flow': s_flow,
        '_ext_hip': ext_hip, '_bar_gap': bar_gap, '_catch_knee': catch_knee,
        '_trunk_max': trunk_max,
    }
    return comps


def _flow_score(feats):
    """动作连贯：bar_h 一阶差分抖动的逆分数。"""
    bh = np.asarray([f['bar_h'] for f in feats])
    if len(bh) < 4:
        return 60.0
    d = np.diff(smooth(bh, 3))
    rms = float(np.sqrt(np.mean(d ** 2)))
    # 慢动作下 rms 偏小：以相对抖动评估，波动越大越不连贯
    return max(0.0, min(100.0, 100.0 - 120.0 * np.clip(rms / (np.max(bh) - np.min(bh) + 1e-3), 0, 0.8)))


def compute_score(components, weights=None):
    """加权合成总分与等级。components 为 score_* 返回的 dict。
    权重之和自动归一；components 缺失的维度自动跳过（剔除项不参与）。"""
    w = weights or DEFAULT_WEIGHTS
    keys = [k for k in w if k in components]
    if not keys:
        return 0.0, '待改进'
    total = sum(components[k] * w[k] for k in keys) / sum(w[k] for k in keys)
    if total >= 85:
        grade = '优秀'
    elif total >= 70:
        grade = '良好'
    elif total >= 55:
        grade = '一般'
    else:
        grade = '待改进'
    return round(total, 1), grade


# ---------------------------------------------------------------- 建议
SUGGESTIONS = {
    'extension': [
        (80, '发力阶段髋角伸展不足（%s°，目标 ≥160°）：顶髋要充分，想象用爆发力把杠铃"顶"起来，而不是用手臂硬拉。'),
        (60, '发力严重不足：下蹲预备时收紧背、肩在杠前，用腿和髋发力蹬伸，杠铃贴腿上升。'),
    ],
    'bar_path': [
        (80, '杠铃离身体过远（水平距离 %s）：拉铃全程让杠铃贴住腿，发力后贴住躯干，想象杠铃沿着身体"擦"上去。'),
        (60, '杠铃轨迹严重外飘：第一拉保持杠铃贴近小腿，第二拉用髋顶杠，肘部向两侧上带。'),
    ],
    'catch': [
        (80, '接铃锁定不稳定（%s）：接铃瞬间主动锁肩、收紧核心，杠铃保持在支撑面正上方后再站起。'),
        (60, '接铃明显晃动/未锁定：先练高抓或架空位锁定，接铃时快速翻肘、杠铃落在锁骨前架。'),
    ],
    'squat': [
        (80, '下蹲深度不够（膝角 %s°）：接铃时要敢往下蹲，用深蹲接铃降低重心，别用半蹲硬撑。'),
        (60, '下蹲过浅或重心偏前：加强接铃深蹲练习，保持脚跟压地、膝盖朝脚尖方向。'),
    ],
    'trunk': [
        (80, '躯干晃动明显（最大倾角 %s°）：全程收紧核心，杠铃上升时保持躯干稳定，别前后晃。'),
        (60, '躯干严重前倾/后仰：降低重量练动作模式，先保证脊柱中立再加重。'),
    ],
    'flow': [
        (80, '动作连贯性一般（%s）：杠铃速度有停顿，发力要一气呵成，减少"先拉后蹬"的脱节。'),
        (60, '动作脱节明显：用悬垂抓/悬垂翻练节奏，第一拉匀速、第二拉爆发，连贯过渡。'),
    ],
    'double_pull': [
        (80, '杠铃上升有停顿/回落（最低速度 %s）：拉铃要一气呵成——一拉匀拉过膝、二拉爆发，杠铃全程不减速不停顿。'),
        (60, '二次拉明显：杠铃上升中断又再次加速，先降重量练高拉节奏，杠铃贴腿连续上升。'),
    ],
    'timing': [
        (80, '杠铃加速时机不准（速度峰值在上升段 %s%% 处，目标约 60%%）：第一拉保持匀速别抢速，过膝后再爆发顶髋。'),
        (60, '发力时机严重偏差：一拉就全力加速会失去三关节顺序，用悬垂翻/悬垂抓练习控制发力节奏。'),
    ],
    'foot': [
        (80, '脚下移动明显（踝位波动 %s）：全程保持站距稳定，杠铃上升与接铃阶段脚都不要挪位。'),
        (60, '脚下大幅移动：先练稳定站姿下的高翻/高抓，脚踩实地面，重心保持在足中。'),
    ],
    'sequence': [
        (80, '伸展顺序不连贯（%s）：发力应先伸髋、再伸膝、最后蹬踝（近端到远端），形成链条式力量传导。'),
        (60, '伸展顺序混乱：用轻重量体会"先顶髋、再伸膝、最后蹬地"的三段节奏，髋发力先于膝踝。'),
    ],
}


def build_suggestions(components, metrics):
    """按分项得分命中建议模板。metrics 提供实测值用于填空。"""
    out = []
    for key, thr_list in SUGGESTIONS.items():
        sc = components.get(key)
        if sc is None:
            continue
        for thr, tmpl in thr_list:
            if sc < thr:
                val = metrics.get(key, '')
                out.append((key, sc, tmpl % (val,) if val else tmpl.split('：')[0] + '。'))
                break
    return out
