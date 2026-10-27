"""姿态提取：用 MediaPipe PoseLandmarker 从视频逐帧提取 33 个人体关键点。"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# 关键点索引（MediaPipe Pose 33 点）
IDX = {
    'nose': 0,
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24,
    'l_knee': 25, 'r_knee': 26,
    'l_ankle': 27, 'r_ankle': 28,
    'l_foot': 31, 'r_foot': 32,
}


class PoseExtractor:
    """逐帧姿态提取器。running_mode=IMAGE，对每一帧独立检测。

    多人入镜时按"髋部位置帧间连续性"锁定主角，避免在不同人之间跳变。
    """

    def __init__(self, model_path='pose_landmarker_full.task', num_poses=2,
                 min_detection_confidence=0.4):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def extract(self, video_path, start=None, end=None):
        """返回 (frames, lms, fps, total)。
        frames: 原图 ndarray (BGR) 列表
        lms: 每帧主角关键点 [(x,y,z,visibility)*33]，无人则 None
        fps: 视频帧率; total: 处理总帧数
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f'无法打开视频: {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        end_frame = total if end is None else min(total, int(end * fps))

        frames, lms = [], []
        prev_hip = None
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if start is not None and idx < int(start * fps):
                continue
            if idx >= end_frame:
                break
            i += 1
            frames.append(fr)
            poses = self._detect(fr)
            chosen = self._choose_main(poses, prev_hip)
            if chosen is not None:
                prev_hip = ((chosen[23][0] + chosen[24][0]) / 2,
                            (chosen[23][1] + chosen[24][1]) / 2)
            lms.append(chosen)
        cap.release()
        lms = self._interpolate(lms)
        return frames, lms, fps, total

    @staticmethod
    def _interpolate(lms):
        """缺失帧线性插值：对检测失败（None）的帧，用前后最近有效帧插值补全。

        插值帧的 visibility 统一压低到 0.5（参与计算但不冒充高置信），
        可靠性剔除机制（样本窗口检查）兜底，避免插值引入的假数据主导打分。
        首尾没有可参考帧的 None 保留原样。
        """
        out = list(lms)
        idxs = [i for i, l in enumerate(out) if l is not None]
        if not idxs or len(idxs) == len(out):
            return out
        for i in range(len(out)):
            if out[i] is not None:
                continue
            before = [j for j in idxs if j < i]
            after = [j for j in idxs if j > i]
            if not before or not after:
                continue
            ia, ib = before[-1], after[0]
            t = (i - ia) / (ib - ia)
            pa = np.array(out[ia], dtype=float)
            pb = np.array(out[ib], dtype=float)
            interp = pa * (1.0 - t) + pb * t
            interp[:, 3] = 0.5  # 低置信标记
            out[i] = interp.tolist()
        return out

    def _detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(img)
        poses = []
        for plm in res.pose_landmarks:
            pts = [(p.x, p.y, p.z, p.visibility) for p in plm]
            poses.append(pts)
        return poses

    @staticmethod
    def _hip(pts):
        return ((pts[23][0] + pts[24][0]) / 2, (pts[23][1] + pts[24][1]) / 2)

    @staticmethod
    def _body_size(pts):
        """用髋-踝距离近似人形大小（近处的人更大）。"""
        h = PoseExtractor._hip(pts)
        a = ((pts[27][0] + pts[28][0]) / 2, (pts[27][1] + pts[28][1]) / 2)
        return ((h[0] - a[0]) ** 2 + (h[1] - a[1]) ** 2) ** 0.5

    def _choose_main(self, poses, prev_hip):
        if not poses:
            return None
        if len(poses) == 1:
            return poses[0]
        if prev_hip is None:
            # 首帧：选画面中最大（最近）的人
            return max(poses, key=self._body_size)
        # 后续帧：选髋部位置离上一帧主角最近的人
        return min(poses, key=lambda p: (self._hip(p)[0] - prev_hip[0]) ** 2
                   + (self._hip(p)[1] - prev_hip[1]) ** 2)
