#!/bin/bash
# 本地启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".env" ]; then
  echo "未找到 .env 文件，请先复制 .env.example："
  echo "  cp .env.example .env"
  echo "  然后编辑填写 OPENAI_API_KEY 等配置"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "创建 Python 虚拟环境..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt -q
fi

# 加载 .env
set -a
source .env
set +a

PORT="${PORT:-8000}"
DEV_RELOAD="${DEV_RELOAD:-0}"
echo "启动后端，端口 $PORT ..."
echo "前端联调：在 resume-annotator-web/.env.local 添加 VITE_API_BASE=http://localhost:$PORT"
echo ""

if [ "$DEV_RELOAD" = "1" ]; then
  exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
fi

exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port "$PORT"
