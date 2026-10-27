# 举重 AI 打分 · 服务器版镜像
# 手机/任意浏览器上传视频 → 服务器分析（ffmpeg 转码 + MediaPipe 姿态 + 十维打分）
FROM python:3.10-slim

# ffmpeg：处理 HEVC/任意编码
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 后端 + 打分管线 + 姿态模型 + 手机端页面
COPY server.py analyze.py scoring.py pose_extractor.py report.py \
     pose_landmarker_full.task server-scorer.html ./

# 可选安全项：访问令牌（默认关闭）
ENV API_TOKEN=""
ENV MAX_MB=200

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
