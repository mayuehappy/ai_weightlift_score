#!/usr/bin/env bash
# 举重 AI 打分 · 服务器一键部署
set -e
cd "$(dirname "$0")"

echo "==> 1/3 构建镜像（首次需下载 Python/MediaPipe 依赖，约 5-10 分钟）"
docker compose build

echo "==> 2/3 启动服务（端口 8000）"
docker compose up -d

echo "==> 3/3 完成"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "✅ 服务已启动："
echo "   手机/浏览器访问：  http://${IP:-服务器公网IP}:8000/"
echo "   本机访问：        http://127.0.0.1:8000/"
echo ""
echo "首次分析会下载 MediaPipe 模型（约 9MB），之后即用即出结果。"
echo "如需设置访问令牌：编辑 docker-compose.yml 的 API_TOKEN 后重新 ./deploy.sh"
