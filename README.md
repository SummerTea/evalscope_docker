# evalscope_docker

自建 EvalScope（ModelScope 官方 LLM 评测框架）Docker 镜像工程。
**镜像内置 17 个核心测试集（除多模态外全维度覆盖），内网完全离线可用，运行时可通过挂载覆盖**；GitHub Actions 自动构建并推送 GHCR + 阿里云 ACR。

## 特性

| 项 | 说明 |
|---|---|
| 服务 | `evalscope service`（Flask Web：任务提交/报告/对比/Arena），端口 9000 |
| 前端 | PyPI wheel 自带构建好的 React SPA，**无需 node 构建阶段** |
| 依赖 | `evalscope[service]` = flask/plotly/perf（aiohttp/uvicorn/numpy），**无 torch**，纯 CPU 可跑 |
| Agent 支持 | 内置 **τ²-bench** 运行依赖（tau2-bench@v0.2.0，airline/retail/telecom 客服域）+ **BFCL-v3**（函数调用，`evalscope[bfcl]`）。τ²-bench **数据**走独立快照路径，运行时按需拉取或挂载（见下） |
| 评测对象 | OpenAI 兼容 API（`EVALSCOPE_BASE_URL` 指向内网 vLLM/SGLang） |
| 指令遵循 | `ifeval` 依赖已内置（`evalscope[ifeval]` = langdetect + nltk + **punkt_tab 词表数据**，构建期预下载到 `~/nltk_data`，内网离线可用） |
| 内置数据 | 17 个测试集（~1-2GB，除多模态外全维度：数学/中文/知识/推理/常识/指令/代码/函数调用） |
| 离线能力 | **纯内网零网络可用**（数据经 MODELSCOPE_CACHE 命中，实测缓存命中 2.3s vs 首次下载 20s） |
| 数据覆盖 | 运行时挂载 `/data/datasets_cache` 即可覆盖内置（全量数据放宿主） |

## 内置测试集（17 个，除多模态外全维度）

| 维度 | benchmark（EvalScope 名） | dataset_id（ModelScope） |
|---|---|---|
| 数学推理 | `gsm8k` / `aime24` / `competition_math` | `AI-ModelScope/gsm8k` / `evalscope/aime24` / `evalscope/competition_math` |
| 中文知识 | `ceval`（52 子集）/ `cmmlu`（67 子集） | `evalscope/ceval` / `evalscope/cmmlu` |
| 通用知识 | `mmlu_pro` / `gpqa_diamond` | `TIGER-Lab/MMLU-Pro` / `AI-ModelScope/gpqa_diamond` |
| 推理 | `bbh`（27 子集）/ `arc` / `agieval`（21 子集） | `evalscope/bbh` / `allenai/ai2_arc` / `opencompass/agieval` |
| 常识 | `hellaswag` / `winogrande` / `truthful_qa` / `commonsense_qa` | `evalscope/hellaswag` / `AI-ModelScope/winogrande_val` / `evalscope/truthful_qa` / `extraordinarylab/commonsense-qa` |
| 指令遵循 | `ifeval` | `opencompass/ifeval` |
| 代码 | `humaneval` | `opencompass/humaneval` |
| **Agent** | **`tau2_bench`**（客服 agent：数据运行时按需拉取或挂载，依赖已内置） | `evalscope/tau2-bench-data` |
| 函数调用 | `bfcl_v3`（17 子集） | `AI-ModelScope/bfcl_v3` |

> 多模态（vlm）/图像/视频/音频测试集不内置（镜像体积约束），需时挂载或按需拉取。

## 快速开始

```bash
# 1. GitHub Actions 自动构建（push main 即触发），产物：
#    ghcr.io/<owner>/evalscope:latest
#    registry.cn-shanghai.aliyuncs.com/luckfu/evalscope:latest

# 2. 或本地构建
docker build -t evalscope:local --build-arg PREFETCH_DATASETS=true .

# 3. 运行（指定被测模型端点）
docker run -d -p 9000:9000 \
  -e EVALSCOPE_BASE_URL=http://10.1.251.230:8000/v1 \
  -e EVALSCOPE_API_KEY=EMPTY \
  -v /nfsdata/eval_platform/outputs:/data/outputs \
  registry.cn-shanghai.aliyuncs.com/luckfu/evalscope:latest

# 4. 打开 Web 界面 http://<host>:9000
```

## 构建参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `EVALSCOPE_VERSION` | `1.11.1` | EvalScope 版本 |
| `PIP_INDEX_URL` | `https://pypi.org/simple` | pip 源（内网换华为云；uv 的 `UV_INDEX_URL` 同源） |
| `PREFETCH_DATASETS` | `true` | 是否预取内置数据集（纯内网可 false，靠挂载） |
| `PREFETCH_CHANNEL` | `ms-first` | 预取通道：`ms-first`（唯一推荐，适配 EvalScope 默认 ModelScope 数据源）。HF 兜底**已弃用**——HF 落盘布局与 EvalScope 运行时 MODELSCOPE_CACHE 布局不兼容，兜底数据必然 miss 联网重下 |
| `HF_ENDPOINT_BUILD` | `https://huggingface.co` | 构建期 HF 端点（仅 `hf-first` 手动兜底场景使用；内置清单不再走 HF） |
| `DATASETS_EXTRA` | 空 | 追加预取数据集（逗号分隔，EvalScope benchmark 名） |

> **依赖安装用 uv**（Rust 实现，pip 10-50x）：`uv pip install` 并行下载 + 极速解析，
> 配合 workflow 的 `cache-from: type=gha` 依赖层缓存，二次构建大幅提速。

## GitHub Actions 构建

`.github/workflows/docker-build.yml`：

- **触发**：push main / tag `v*` / 手动 `workflow_dispatch`
- **产物**：`ghcr.io/<owner>/evalscope:{latest|sha7|版本}` + `registry.cn-shanghai.aliyuncs.com/luckfu/evalscope:{latest|sha7|版本}`
- **平台**：amd64 + arm64（buildx + QEMU）
- **数据**：默认 `PREFETCH_DATASETS=true`、`ms-first` 通道
- **ACR 兼容**：`provenance: false, sbom: false`（阿里云 ACR 不认 oci.empty manifest）
- **凭据**：GHCR 用 `GITHUB_TOKEN`；阿里云 ACR 用仓库 Secrets `ALIYUN_REGISTRY_USERNAME` / `ALIYUN_REGISTRY_PASSWORD`

> ⚠️ **官方 runner 磁盘 ~14GB**：内置数据仅限精选核心集 ~1-2GB；
> 全量数据（20-40GB+）请运行时挂载 `/data/datasets_cache` 覆盖。

## 数据集机制（重要，实测验证）

EvalScope 数据集加载（源码实证 `evalscope/api/dataset/hub.py` + 实测）：

```
dataset_hub=MODELSCOPE  -> MsDataset.load(id) 不传 cache_dir，命中 MODELSCOPE_CACHE env 的默认路径
                           <cache>/datasets/<org>___<name>/...（arrow 缓存）
dataset_hub=HUGGINGFACE -> datasets.load_dataset(id)，命中 HF_HOME（镜像内 /data/datasets_cache/hf_home）
dataset_id 是本地路径    -> os.path.exists 命中即本地读取，零网络
```

**离线命中的关键**（踩坑记录）：
1. 构建期预下载必须用 **`MsDataset.load`**（与运行时同调用路径），不能 `dataset_snapshot_download`
2. 预下载**不能传 `cache_dir`**——必须依赖 `MODELSCOPE_CACHE` env，让落盘布局（`<cache>/datasets/<org>___<name>/`）与运行时完全一致
3. 实测：构建期缓存 + 运行时同路径 → 第二次加载 2.3s（命中）；路径不一致 → 永远 miss 联网下载

- **内置**：构建期 `scripts/prefetch_datasets.py`（设置 `MODELSCOPE_CACHE=/data/datasets_cache` 后 `MsDataset.load` 预下载各 split，覆盖 test/validation/val/dev/train）
- **覆盖**：运行时 `-v <host>:/data/datasets_cache` 挂载即整体覆盖内置（宿主数据优先）
- **离线**：纯内网 `HF_HUB_OFFLINE=1` + 内置数据，完全零网络
- **不用 HF 兜底**：HF `snapshot_download` 落盘 `<hf_home>/datasets/<org>__<name>/`，而 EvalScope 运行时查 `MODELSCOPE_CACHE/datasets/<org>___<name>/`（hub.py 不传 cache_dir）——布局不兼容必然 miss，故内置清单只走 MS 通道

自定义预取清单：`--datasets "gsm8k,ceval"`（EvalScope benchmark 名）或 `--datasets-file list.txt`。

## 内网部署

```bash
docker compose up -d     # docker-compose.yml 已配好挂载与环境变量
```

半内网（可访问 hf-mirror）：设 `HF_ENDPOINT=https://hf-mirror.com`，运行时按需下载未内置数据。
纯内网：内置 17 测试集开箱即用；更多数据挂载 `/data/datasets_cache` 覆盖 + `HF_HUB_OFFLINE=1`。
