# 简历批注后端 MVP

FastAPI 后端，封装 `resume-review-annotator-v2.py` 处理链，提供任务创建、状态查询、结果读取和文件下载 API。

---

## 目录结构

```
resume-annotator-backend/
├── main.py              # FastAPI 入口，CORS、路由挂载、全局错误处理
├── api/
│   ├── auth.py          # 登录接口（JWT，MVP 固定账号）
│   └── tasks.py         # 任务 CRUD + 下载接口
├── services/
│   ├── storage.py       # 文件存储抽象（本地磁盘，可扩展 S3/R2）
│   ├── task_service.py  # 任务元数据读写（SQLite）
│   ├── processor.py     # 处理链编排（AI review → render → report）
│   ├── providers.py     # AI provider 抽象（OpenAI / Claude）
│   └── adapter.py       # 原始 JSON → 前端格式 adapter
├── models/
│   ├── db.py            # SQLite 初始化
│   └── schemas.py       # Pydantic 模型
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 本地启动

### 1. 前置依赖

```bash
# macOS
brew install poppler tesseract tesseract-lang

# 验证
pdftotext --version
pdftoppm --version
tesseract --version
```

### 2. Python 环境

```bash
cd resume-annotator-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 环境变量

```bash
cp .env.example .env
# 编辑 .env
```

**直连 OpenAI 官方：**
```
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
JWT_SECRET=any-random-long-string
ADMIN_EMAIL=you@example.com
ADMIN_PASSWORD=yourpassword
```

**通过 OneAPI / 中转代理：**
```
AI_PROVIDER=openai
OPENAI_BASE_URL=https://your-oneapi.com   # 中转地址，不含 /v1
OPENAI_API_KEY=sk-your-oneapi-key         # 中转平台的 Key
OPENAI_MODEL=gpt-4o-mini
JWT_SECRET=any-random-long-string
ADMIN_EMAIL=you@example.com
ADMIN_PASSWORD=yourpassword
```

**使用 Claude：**
```
AI_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-opus-4-6
JWT_SECRET=any-random-long-string
ADMIN_EMAIL=you@example.com
ADMIN_PASSWORD=yourpassword
```

### 4. 启动

```bash
# 确保激活了 .venv
uvicorn main:app --reload --port 8000
```

后端在 http://localhost:8000，Swagger 文档在 http://localhost:8000/docs

启动日志里会打印当前使用的 provider 和 endpoint，例如：
```
Using OpenAI provider (model=gpt-4o-mini, base=https://your-oneapi.com)
```

### 5. 验证 OneAPI 是否真正生效

**方法 1 — 看启动日志：**
```
INFO  services.providers: Using OpenAI provider (model=gpt-4o-mini, base=https://your-oneapi.com)
```
如果 base 是 `https://api.openai.com`，说明 `OPENAI_BASE_URL` 没有生效，检查 `.env` 是否保存。

**方法 2 — 直接测试 OneAPI 连通性：**
```bash
curl -s https://your-oneapi.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | python3 -m json.tool | head -20
```
返回模型列表说明 Key 和 Base URL 都正确。

**方法 3 — 创建一个任务，看 failReason：**
失败时 `GET /api/tasks/:id` 会返回完整的 `failReason`，包含 HTTP 状态码和完整响应体，可以直接判断是 401（Key 错）、404（路径错）还是 429（限流）。

### 6. 前端联调

在 `resume-annotator-web/.env.local` 中设置：

```
VITE_API_BASE=http://localhost:8000
```

重启前端 `npm run dev` 后即可对接真实后端。

---

## AI Provider 切换

通过 `.env` 中的 `AI_PROVIDER` 环境变量切换大模型：

| 值 | 使用的 API | 必填 env var |
|---|---|---|
| `openai`（默认）| OpenAI Responses API `/v1/responses` | `OPENAI_API_KEY` |
| `claude` | Anthropic Messages API `/v1/messages` + tool_use | `ANTHROPIC_API_KEY` |

**切换为 Claude：**
```bash
# .env
AI_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL=claude-opus-4-6   # 或 claude-sonnet-4-6
```

**架构说明：**
- 只有 `review-match`（AI 分析）步骤走 provider
- `render-pdf`（渲染批注 PDF）和 `report-md`（生成报告）是纯本地操作，两种 provider 共用同一套本地脚本，不额外消耗 token

---

## 接口清单

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/auth/login | 登录，返回 JWT |
| GET  | /api/tasks | 任务列表 |
| GET  | /api/tasks/:id | 单个任务状态 |
| POST | /api/tasks | 创建任务（上传 PDF + JD 图片） |
| GET  | /api/tasks/:id/result | 获取分析结果 |
| GET  | /api/tasks/:id/download/annotated_pdf | 下载批注 PDF |
| GET  | /api/tasks/:id/download/report | 下载分析报告（HTML） |
| GET  | /health | 健康检查 |

---

## 存储方案

- **元数据**：SQLite，路径 `$DATA_DIR/tasks.db`（默认 `./data/tasks.db`）
- **上传文件**：`$DATA_DIR/uploads/{task_id}/`
- **输出产物**：`$DATA_DIR/outputs/{task_id}/`

| 文件 | 路径 | 说明 |
|------|------|------|
| result.json | outputs/{id}/result.json | 原始 AI 结果（内部使用） |
| annotated.pdf | outputs/{id}/annotated.pdf | 批注版 PDF（对外下载） |
| report.md | outputs/{id}/report.md | Markdown 报告（中间格式） |
| report.html | outputs/{id}/report.html | HTML 报告（对外下载） |

---

## 部署到 Render

### 方式一：Docker 部署（推荐）

1. 在 Render 新建 Web Service，选择「Docker」
2. 设置 Root Directory 为 `resume-annotator-backend`
3. 添加环境变量：
   - `OPENAI_API_KEY`
   - `JWT_SECRET`（随机长字符串）
   - `ADMIN_EMAIL` / `ADMIN_PASSWORD`
   - `FRONTEND_ORIGIN`（前端部署 URL，如 `https://your-app.pages.dev`）
   - `DATA_DIR=/data`
4. 挂载 Persistent Disk 到 `/data`（在 Render 付费计划中选 Disk）
5. 部署后在前端设置 `VITE_API_BASE=https://your-backend.onrender.com`

### 今天最快上线的推荐结构

- 前端：Cloudflare Pages，绑定 `https://erenlab.cn`
- 后端：Render Docker Web Service，绑定 `https://api.erenlab.cn`
- 前端环境变量：
  - `VITE_API_BASE=https://api.erenlab.cn`
- 后端环境变量：
  - `FRONTEND_ORIGIN=https://erenlab.cn`
  - `FRONTEND_ORIGINS=https://www.erenlab.cn`
  - `DATA_DIR=/data`
  - `JWT_SECRET=...`
  - `ADMIN_EMAIL=...`
  - `ADMIN_PASSWORD=...`
  - `OPENAI_API_KEY=...`（或你自己的中转配置）

仓库已提供：

- `Dockerfile`
- `.dockerignore`
- `render.yaml`

如果直接用 Render Blueprint / Git 部署，重点确认：

1. Web Service 使用 Docker runtime
2. 挂载 Persistent Disk 到 `/data`
3. 自定义域名填写 `api.erenlab.cn`
4. Cloudflare DNS 为 `api` 子域名添加 CNAME 指向 Render 分配的目标
5. Cloudflare Pages 中把 `VITE_API_BASE` 改为 `https://api.erenlab.cn`

### 方式二：Python 直接部署

1. 在 Render 新建 Web Service，选择「Python」
2. Build Command：
   ```
   pip install -r requirements.txt
   apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-chi-sim || true
   ```
   > 注意：Render 默认环境不包含 `tesseract`/`poppler-utils`，需要 Dockerfile 方式确保系统依赖。

### Render 系统依赖说明

| 依赖 | 用途 | 安装 |
|------|------|------|
| `poppler-utils` | `pdftotext`, `pdftoppm` | Dockerfile 中 `apt-get install poppler-utils` |
| `tesseract-ocr` | OCR 备用文本提取 | Dockerfile 中 `apt-get install tesseract-ocr tesseract-ocr-chi-sim` |
| `Pillow` | PDF 页面图像处理 | `pip install Pillow`（已在 requirements.txt） |

---

## 替换正式用户系统

当前 MVP 使用环境变量里的固定账号。替换方法：

1. 在 `api/auth.py` 的 `_USERS` dict 改为数据库查询
2. 在 `models/db.py` 中新增 `users` 表
3. 保留 `_make_token` 和 `_decode_token` 不变（JWT 逻辑不用动）

---

## 当前不包含的能力（后续迭代）

- 多用户系统（当前 MVP 为单账号）
- 云对象存储（当前为本地磁盘，`storage.py` 已做抽象，扩展替换 `save_upload` 和 `get_download_path` 即可）
- 任务队列（当前为 ThreadPoolExecutor，5人以内够用）
- 速率限制和并发控制
- 任务重试机制
