# evalscope_docker

自建 EvalScope（ModelScope 官方 LLM 评测框架）Docker 镜像工程。
**镜像内置核心数据集，运行时可通过挂载覆盖**；GitHub Actions 自动构建并推送 GHCR。

## 特性

| 项 | 说明 |
|---|---|
| 服务 | `evalscope service`（Flask Web：任务提交/报告/对比/Arena），端口 9000 |
| 前端 | PyPI wheel 自带构建好的 React SPA，**无需 node 构建阶段** |
| 依赖 | `evalscope[service]` = flask/plotly/perf（aiohttp/uvicorn/numpy），**无 torch**，纯 CPU 可跑 |
| 评测对象 | OpenAI 兼容 API（`EVALSCOPE_BASE_URL` 指向内网 vLLM/SGLang） |
| 内置数据 | 核心数据集 ~1GB（tinyBenchmarks 冒烟层 + MMLU/GSM8K/C-Eval/CMMLU/HumanEval/wikitext） |
| 数据覆盖 | 运行时挂载 `/data/datasets_cache` 即可覆盖内置（全量数据放宿主） |

## 快速开始

```bash
# 1. GitHub Actions 自动构建（push main 即触发），产物在 GHCR：
#    ghcr.io/<owner>/evalscope:latest

# 2. 或本地构建
docker build -t evalscope:local --build-arg PREFETCH_DATASETS=true .

# 3. 运行（指定被测模型端点）
docker run -d -p 9000:9000 \
  -e EVALSCOPE_BASE_URL=http://10.1.251.230:8000/v1 \
  -e EVALSCOPE_API_KEY=EMPTY \
  -v /nfsdata/eval_platform/outputs:/data/outputs \
  ghcr.io/<owner>/evalscope:latest

# 4. 打开 Web 界面 http://<host>:9000
```

## 构建参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `EVALSCOPE_VERSION` | `1.11.1` | EvalScope 版本 |
| `PIP_INDEX_URL` | `https://pypi.org/simple` | pip 源（内网换华为云；uv 的 `UV_INDEX_URL` 同源） |
| `PREFETCH_DATASETS` | `true` | 是否预取核心数据集（纯内网可 false，靠挂载） |
| `PREFETCH_CHANNEL` | `hf-first` | 预取通道：`hf-first`（Actions 公网，HF 直连）/ `ms-first`（内网，ModelScope 优先） |
| `HF_ENDPOINT_BUILD` | `https://huggingface.co` | 构建期 HF 端点（内网构建换 hf-mirror） |
| `DATASETS_EXTRA` | 空 | 追加预取数据集（逗号分隔） |

> **依赖安装用 uv**（Rust 实现，pip 10-50x）：`uv pip install` 并行下载 + 极速解析，
> 配合 workflow 的 `cache-from: type=gha` 依赖层缓存，二次构建大幅提速。

## GitHub Actions 构建

`.github/workflows/docker-build.yml`：

- **触发**：push main / tag `v*` / 手动 `workflow_dispatch`
- **产物**：`ghcr.io/<owner>/evalscope:{latest|sha7|版本}`
- **平台**：amd64 + arm64（buildx + QEMU）
- **数据**：默认 `PREFETCH_DATASETS=true`、`hf-first` 通道（Actions 公网 HF 直连最顺）

> ⚠️ **官方 runner 磁盘 ~14GB**：内置数据仅限精选核心集 ~1GB；
> 全量数据（20-40GB+）请运行时挂载 `/data/datasets_cache` 覆盖。

## 数据集机制（重要）

EvalScope 数据集加载（源码实证 `evalscope/api/dataset/hub.py` + `default_data_adapter.py`）：

```
dataset_hub=MODELSCOPE  -> MsDataset.load(id)，命中 MODELSCOPE_CACHE（镜像内 /data/datasets_cache）
dataset_hub=HUGGINGFACE -> datasets.load_dataset(id)，命中 HF_HOME（镜像内 /data/datasets_cache/hf_home）
dataset_id 是本地路径    -> os.path.exists 命中即本地读取，零网络
```

- **内置**：构建期 `scripts/prefetch_datasets.py` 把核心集灌进 `/data/datasets_cache`
- **覆盖**：运行时 `-v <host>:/data/datasets_cache` 挂载即整体覆盖内置（推荐把宿主全量数据挂载进来）
- **离线**：纯内网设 `HF_HUB_OFFLINE=1`，数据全部来自挂载卷

自定义预取清单：`--datasets "id1,id2"` 或 `--datasets-file list.txt`。

## 内网部署

```bash
docker compose up -d     # docker-compose.yml 已配好挂载与环境变量
```

半内网（可访问 hf-mirror）：设 `HF_ENDPOINT=https://hf-mirror.com`，运行时按需下载。
纯内网：`PREFETCH_DATASETS=false` 构建（或接受内置精选集），全量数据挂载覆盖 + `HF_HUB_OFFLINE=1`。
