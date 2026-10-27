"""举重 AI 打分 · 服务器版后端（Docker 一键部署）。

手机/任何浏览器上传整视频 → 服务器 ffmpeg 转码 + MediaPipe 姿态 → 十维打分 → 返回结果。

接口:
    POST /api/analyze   multipart: file=<视频>, lift=snatch|clean
                        -> 打分 JSON（含标注视频与关键帧的媒体 URL）
    GET  /              -> 手机端页面（server-scorer.html，同源调用 API）

安全:
    - 文件大小上限 MAX_MB（默认 200）
    - 仅允许常见视频扩展名
    - jobs 目录定时清理（3 天前自动删除）
    - 可选共享密钥: 设置环境变量 API_TOKEN 后，请求需带 ?token=<TOKEN> 或 X-Api-Token 头
"""
import argparse
import json
import os
import shutil
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import analyze

BASE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE, 'jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

MAX_MB = int(os.environ.get('MAX_MB', '200'))
MAX_BYTES = MAX_MB * 1024 * 1024
API_TOKEN = os.environ.get('API_TOKEN', '').strip()
JOB_TTL_SEC = 3 * 24 * 3600  # jobs 保留 3 天

app = FastAPI(title='举重 AI 打分服务端')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)


@app.on_event('startup')
def _startup():
    cleanup_jobs()


def check_token(request):
    if not API_TOKEN:
        return
    tok = request.query_params.get('token') or request.headers.get('x-api-token')
    if tok != API_TOKEN:
        raise HTTPException(status_code=401, detail='缺少或错误的访问令牌')


def cleanup_jobs():
    now = time.time()
    for name in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, name)
        try:
            if now - os.path.getmtime(p) > JOB_TTL_SEC:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


@app.get('/')
def index():
    return HTMLResponse(open(os.path.join(BASE, 'server-scorer.html'), encoding='utf-8').read())


@app.get('/media/{job}/{file}')
def media(job: str, file: str):
    if '/' in file or '\\' in file or not file.replace('.', '').replace('_', '').replace('-', '').isalnum():
        raise HTTPException(status_code=400, detail='非法文件名')
    p = os.path.join(JOBS_DIR, job, file)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail='文件不存在或已过期')
    return FileResponse(p)


@app.post('/api/analyze')
async def api_analyze(request: Request, file: UploadFile = File(...), lift: str = Form('clean')):
    check_token(request)
    lift = lift.strip().lower()
    if lift not in ('snatch', 'clean'):
        raise HTTPException(status_code=400, detail='lift 仅支持 snatch（抓举）或 clean（高翻）')
    name = (file.filename or 'upload.mp4').lower()
    if not name.endswith(('.mp4', '.mov', '.avi', '.webm', '.m4v', '.mkv', '.3gp')):
        raise HTTPException(status_code=400, detail='不支持的文件格式，请上传 mp4/mov/avi/webm/m4v 等视频')

    job = uuid.uuid4().hex[:10]
    jobdir = os.path.join(JOBS_DIR, job)
    os.makedirs(jobdir, exist_ok=True)
    vp = os.path.join(jobdir, 'upload' + os.path.splitext(name)[1])
    size = 0
    with open(vp, 'wb') as f:
        while True:
            chunk = file.file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                shutil.rmtree(jobdir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f'文件过大（>{MAX_MB}MB），请压缩后上传')
            f.write(chunk)
    if size < 4096:
        shutil.rmtree(jobdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail='文件为空或太小')

    try:
        analyze.analyze(vp, lift=lift, out=jobdir, annotate=True)
    except ValueError as e:
        shutil.rmtree(jobdir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        shutil.rmtree(jobdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f'分析失败：{e}')

    stem = os.path.splitext(os.path.basename(vp))[0]
    rp = os.path.join(jobdir, f'{stem}_{lift}_result.json')
    with open(rp, encoding='utf-8') as f:
        result = json.load(f)
    result['media'] = {
        'annotated': f'/media/{job}/{stem}_{lift}_annotated.mp4',
        'keyframes': {k: f'/media/{job}/phase_{k}.jpg'
                      for k in ('setup', 'ext', 'catch', 'stand')},
    }
    return JSONResponse(result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='举重 AI 打分服务器版')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    print(f'举重 AI 打分服务已启动：http://{args.host}:{args.port}/', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
