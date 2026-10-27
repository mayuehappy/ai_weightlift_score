# 举重 AI 打分 · 服务器版（手机/任何浏览器可用）

上传视频 → 服务器分析（ffmpeg 转码 + MediaPipe 姿态 + 十维规则打分）→ 返回总分、十维评分、四阶段关键帧、标注视频与改进建议。
**彻底解决手机端兼容问题**：任何格式（含 iPhone HEVC）、任何浏览器（含微信内）都能用。

---

## 一、你需要一台公网服务器

### 最省钱的购买建议（国内，手机访问快）

| 方案 | 配置 | 价格参考 | 说明 |
|---|---|---|---|
| 学生机（推荐） | 2核2G | 约 99 元/年 | 阿里云/腾讯云学生认证都有；够用 |
| 轻量应用服务器 | 2核2G | 约 24-40 元/月 | 新用户常有 1-3 折活动 |
| 轻量应用服务器 | 2核4G | 约 70-100 元/月 | 分析更快、更稳（推荐预算允许时） |

- 系统选 **Ubuntu 22.04**，地域选离你近的（北京/上海/广州）
- 带宽 3Mbps 起即可（6MB 视频上传约 3-5 秒）
- 安全组/防火墙放行 **8000 端口**（或 80/443）

### 为什么需要它
纯前端版（GitHub Pages）在部分手机上受浏览器限制；服务器版把计算搬到服务器，手机只是上传与展示，**100% 兼容**。

---

## 二、部署（一条命令）

在服务器上执行（需已装 Docker，或先执行 install-docker 命令）：

```bash
# 1. 安装 Docker（Ubuntu）
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker && sudo systemctl start docker

# 2. 获取部署包（任选）
git clone https://github.com/mayuehappy/ai_weightlift_score.git && cd ai_weightlift_score/server
# 或把本 server/ 目录上传到服务器

# 3. 一键部署（首次构建约 5-10 分钟，之后秒起）
sudo bash deploy.sh
```

完成后手机访问：`http://<服务器公网IP>:8000/`

### 部署脚本做了什么
```bash
docker compose build   # 构建镜像（Python + MediaPipe + ffmpeg）
docker compose up -d   # 启动服务（端口 8000，自动重启）
```

---

## 三、使用

1. 手机浏览器（或微信）打开 `http://<服务器IP>:8000/`
2. 点「选择视频」→ 选任意格式的视频（mp4/mov/avi/webm…）
3. 选动作类型：高翻 clean / 抓举 snatch
4. 点「开始分析」→ 显示上传进度 → 服务器分析（约 20-60 秒）→ 出结果
5. 结果含：总分与等级、十维评分表、四阶段关键帧、标注视频、可靠性与改进说明

---

## 四、可选加固

### 1. 访问令牌（防陌生人滥用）
编辑 `docker-compose.yml`：
```yaml
environment:
  API_TOKEN: "你的随机字符串"   # 手机访问时 URL 加 ?token=xxx
```
然后重新 `sudo bash deploy.sh`。

### 2. 域名 + HTTPS（强烈建议，尤其要分享给别人）
- 买域名后解析到服务器 IP
- 用 Caddy 自动 HTTPS（示例 `Caddyfile`）：
```
lift.example.com {
  reverse_proxy 127.0.0.1:8000
}
```
- 之后链接变为 `https://lift.example.com`，更可信、微信内更友好

### 3. 上传限制
默认单文件 ≤200MB（`MAX_MB` 可调）；结果保留 3 天自动清理（`JOB_TTL`）。

---

## 五、架构

```
手机/任意浏览器 ──上传视频──▶ 服务器:8000 (FastAPI)
                                ├─ ffmpeg 转码（任意编码→可处理）
                                ├─ MediaPipe 姿态提取（33 关键点）
                                ├─ 十维规则打分（已校准）
                                └─ 返回 JSON + 标注视频 + 关键帧
手机 ◀──渲染结果───
```

---

## 六、验证方式

- 本地验证：`python3 server.py` 后 `curl -F "file=@test.mp4" -F "lift=clean" http://127.0.0.1:8000/api/analyze`
- 返回 `{total_score, grade, components, media:{annotated, keyframes}}` 即成功
- 参考实测：6.9s HEVC 视频，12.5 秒完成分析，总分 87.1 / 优秀

评分供训练参考，非竞赛裁判标准。
