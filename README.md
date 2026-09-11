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

---

## 一、内置数据集（17 个，除多模态外全维度）

### 1.1 一览表

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

### 1.2 能力说明（来源：EvalScope 1.11.1 adapter 源码实证）

| benchmark | 评估能力 | 规模 / 格式 | 子集与 split | few-shot | 备注 |
|---|---|---|---|---|---|
| `gsm8k` | 小学数学推理 | 8.5k 题，main/socratic | subset=`main`；eval=test, train=train | 4-shot | 经典数学题，答案数字 |
| `aime24` | 竞赛数学（AIME-2024） | 30 题 | eval=test | 0-shot | 高难度竞赛题 |
| `competition_math` | 竞赛数学（MATH） | 12.5k 题，5 难度等级 | 单 config；按 `level` 列分组 Level 1-5；eval=test, train=train | 4-shot | reformat 数据集，输出 `\boxed{}` 答案 |
| `ceval` | 中文知识（C-Eval） | 52 科目，~13k 题 | 52 子集（科目）；eval=val, train=dev | 5-shot | 小学到专业级中文选择题 |
| `cmmlu` | 中文知识（CMMLU） | 67 科目 | 单 config；按 `category` 列分组；eval=test, train=dev | 0-shot | 67 中文领域选择题 |
| `mmlu_pro` | 通用知识（MMLU-Pro） | 14 学科，~12k 题 | 14 子集；eval=test, train=validation | 5-shot | 多学科选择题，干扰项更多 |
| `gpqa_diamond` | 科学推理（GPQA-Diamond） | 198 题 | eval=train（仅 train split） | 0-shot | 专家级科学多选题 |
| `bbh` | 综合推理（BBH） | 27 子集，~6.5k 题 | 27 子集；eval=test | 3-shot | 多选+自由格式混合推理 |
| `arc` | 科学推理（ARC） | ARC-Easy/Challenge 2 子集 | 2 子集；eval=test, train=train | 0-shot | 小学科学题，Challenge 更硬 |
| `agieval` | 综合推理（AGIEval，中英） | 21 子集 | 21 子集；eval=test, train=dev | 0-shot | 高考/LSAT/SAT/GRE 风格 |
| `hellaswag` | 常识推理（HellaSwag） | 10k 样本 | eval=validation | 0-shot | 句子补全常识题 |
| `winogrande` | 常识消歧（Winogrande） | 1.3k 样本 | eval=validation | 0-shot | 代词消歧 |
| `truthful_qa` | 事实性（TruthfulQA） | 817 题，MC1/MC2 | subset=`multiple_choice`；eval=validation | 0-shot | 检测模型是否传播误解，`multiple_correct=True` 切 MC2 |
| `commonsense_qa` | 常识问答（CommonsenseQA） | 9.7k 题 | eval=validation | 0-shot | 常识多选题 |
| `ifeval` | 指令遵循（IFEval） | 541 题，25 类指令 | eval=train | 0-shot | 逐指令可验证（依赖 langdetect+nltk punkt_tab，已内置） |
| `humaneval` | 代码生成（HumanEval） | 164 题 | subset=`openai_humaneval`；eval=test | 0-shot | Python 函数补全 |
| `bfcl_v3` | 函数调用（BFCL-v3） | 17 子集，2k+ 场景 | 17 子集（simple/multiple/parallel/live/multi_turn…）；eval=train | 0-shot | 函数调用/工具使用（依赖 bfcl-eval+soundfile，已内置） |

---

## 二、快速开始

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

---

## 三、使用指南

### 3.1 Web 界面

- **入口**：`http://<host>:9000`
- **评测对象**：容器启动时 `EVALSCOPE_BASE_URL` 指向的模型（也可界面内填 API 端点）
- **内置能力**：任务提交、报告查看、ComparePage（双模型/双方案 delta 对比）、PerfComparePage（性能对比）、Arena

**任务可见性机制**（1.11.1 行为，源码实证）：
- **dashboard / Reports 页只显示已完成的报告**——磁盘扫描 `<outputs>/<run_id>/reports/<model>/`，运行中任务不显示
- **运行中任务的进度/日志**在**提交任务的页面**实时看（5s 轮询 `/api/v1/eval/progress`，前端内存态，**刷新页面即丢失**）
- 所以：长任务期间别刷新提交页；跑完的报告刷新 dashboard 即可见
- 多数据集提交时注意：**已完成的子集报告会先落盘**，dashboard 会随进度逐个出现

**Web 提交注意事项**：
- **模型名不要带尾随空格**（如 `Qwen3.8-27B`，别填 `Qwen3.8-27B `，会 404）
- **批大小（eval_batch_size）**：填默认 `8` 即可。实测并发 1/8/32 总时长几乎无差异（瓶颈在 vLLM 单请求推理 + 数据集加载），调大不会更快
- **limit 是全局的**：一次提交多数据集时对所有数据集生效（ceval 52 子集 × limit 会显著放大耗时）

**时间参考**（Qwen3.8-27B @ A800 vLLM，实测）：
- gsm8k limit=10 ≈ 17s；bfcl_v3 limit=5 ≈ 27s；mmlu_pro limit=10 ≈ 22s；ifeval limit=10 ≈ 21s
- bbh limit=10 ≈ 107s（27 子集）；humaneval limit=10 ≈ 167s（代码长输出）
- **ceval limit=5 ≈ 6 分钟**（52 子集 × 固定加载开销，时间黑洞）
- **aime24/competition_math**：27B 长数学推理链，单条 46-73s 且打满 max_tokens，易触发 openai 客户端 600s 读超时——建议 limit≤2 或从 30 分钟测试集里剔除

### 3.2 CLI 评测（OpenAI 兼容 API 模式）

```bash
# 单数据集冒烟（limit 少量样本快速验证）
docker exec evalscope evalscope eval \
  --model-id Qwen3.8-27B --eval-type openai_api \
  --api-url http://10.1.251.230:8000/v1 --api-key EMPTY \
  --datasets gsm8k --limit 5 \
  --no-timestamp --work-dir /data/outputs/smoke_gsm8k

# 多数据集一次评测（17 内置测试集全量，limit=5 约 15-20 分钟）
docker exec evalscope evalscope eval \
  --model-id Qwen3.8-27B --eval-type openai_api \
  --api-url http://10.1.251.230:8000/v1 --api-key EMPTY \
  --datasets gsm8k aime24 competition_math ceval cmmlu mmlu_pro gpqa_diamond \
             bbh arc agieval hellaswag winogrande truthful_qa commonsense_qa \
             ifeval humaneval bfcl_v3 \
  --limit 5 \
  --no-timestamp --work-dir /data/outputs/full_smoke

# 全量正式评测（不加 --limit；ceval 52 子集 / aime24 长推理链会较久）
```

**CLI 结果让 Web 页面可见的关键**（`--no-timestamp` + work-dir 命名规则）：

页面（dashboard / Reports）扫描磁盘结构 `<outputs>/<run_id>/reports/<model>/`，CLI 必须：

1. **加 `--no-timestamp`**——否则报告落在 `<work_dir>/<timestamp>/reports/`，多套一层时间戳目录，扫描器找不到
2. **`--work-dir` 必须是 `/data/outputs/` 下的命名子目录**（如 `/data/outputs/my_task`）——不能直接写 `/data/outputs`（那样报告落到 `/data/outputs/reports/`，缺 run_id 层，同样扫不到）
3. 每次用**新的任务名**（`--no-timestamp` 后同名目录会直接覆盖旧结果）

满足上述条件，CLI 跑完刷新 dashboard 即可看到该任务。**Web 提交的任务**（`/data/outputs/<task_id>/reports/`）天然符合此结构，无需额外参数。

### 3.3 数据集与离线验证

```bash
# 查看内置数据集缓存（MS 布局：<cache>/datasets/<org>___<name>/）
docker exec evalscope ls /data/datasets_cache/datasets/

# 验证某个数据集离线命中（应 <5s，无网络下载日志）
docker exec evalscope python3 -c \
  "from modelscope.msdatasets import MsDataset; d=MsDataset.load(dataset_name='AI-ModelScope/gsm8k', split='test'); print('rows:', len(d))"

# 断网离线（纯内网）：容器内加载走 MODELSCOPE_CACHE，零网络
```

---

## 四、容器运维

```bash
# 拉取最新镜像（ACR 间歇性拒绝，失败重试即可）
docker pull registry.cn-shanghai.aliyuncs.com/luckfu/evalscope:latest

# 启动（指定被测模型端点）
docker run -d --name evalscope -p 9000:9000 \
  -e EVALSCOPE_BASE_URL=http://10.1.251.230:8000/v1 \
  -e EVALSCOPE_API_KEY=EMPTY \
  -v /nfsdata/eval_platform/outputs:/data/outputs \
  registry.cn-shanghai.aliyuncs.com/luckfu/evalscope:latest

# 进入容器 / 看日志 / 停止
docker exec -it evalscope bash
docker logs -f evalscope
docker rm -f evalscope
```

## 五、内网部署

```bash
docker compose up -d     # docker-compose.yml 已配好挂载与环境变量
```

半内网（可访问 hf-mirror）：设 `HF_ENDPOINT=https://hf-mirror.com`，运行时按需下载未内置数据。
纯内网：内置 17 测试集开箱即用；更多数据挂载 `/data/datasets_cache` 覆盖 + `HF_HUB_OFFLINE=1`。

---

## 六、构建镜像

### 6.1 构建参数

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

### 6.2 GitHub Actions 构建

`.github/workflows/docker-build.yml`：

- **触发**：push main / tag `v*` / 手动 `workflow_dispatch`
- **产物**：`ghcr.io/<owner>/evalscope:{latest|sha7|版本}` + `registry.cn-shanghai.aliyuncs.com/luckfu/evalscope:{latest|sha7|版本}`
- **平台**：amd64 + arm64（buildx + QEMU）
- **数据**：默认 `PREFETCH_DATASETS=true`、`ms-first` 通道
- **ACR 兼容**：`provenance: false, sbom: false`（阿里云 ACR 不认 oci.empty manifest）
- **凭据**：GHCR 用 `GITHUB_TOKEN`；阿里云 ACR 用仓库 Secrets `ALIYUN_REGISTRY_USERNAME` / `ALIYUN_REGISTRY_PASSWORD`

> ⚠️ **官方 runner 磁盘 ~14GB**：内置数据仅限精选核心集 ~1-2GB；
> 全量数据（20-40GB+）请运行时挂载 `/data/datasets_cache` 覆盖。

---

## 七、被测模型服务（vLLM）启动参数

EvalScope 通过 OpenAI 兼容 API 评测（`EVALSCOPE_BASE_URL` 指向 vLLM/SGLang）。**不同测试集对 vLLM 服务端有不同要求**，参数不全会导致评测失败（实测踩坑）。

### 7.1 推荐启动命令（Qwen3.8-27B @ A800 实测）

```bash
docker run -d --name vllm --gpus device=1 -p 8000:8000 \
  -v /nfsdata/models:/models \
  -v /nfsdata/vllm_cache:/root/.cache/vllm \
  -e HF_HOME=/tmp/hf-cache \
  harbor.cloud.com/vllm/vllm-openai:v0.28.0 \
  /models/Qwen3.8-27B --host 0.0.0.0 --port 8000 \
  --served-model-name Qwen3.8-27B \
  --enable-prefix-caching --no-enable-log-requests \
  --gpu-memory-utilization 0.92 --max-model-len 131072 --max-num-seqs 256 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

### 7.2 各参数的作用与必要性（按测试集）

| 参数 | 必要性 | 支撑的测试集 |
|---|---|---|
| `--enable-auto-tool-choice --tool-call-parser <parser>` | **必须**（bfcl_v3 等函数调用） | `bfcl_v3`（agent 工具调用） |
| `--enable-prefix-caching` | 强烈建议（多请求共享前缀加速，实测显著提速） | 全部（尤其选择题多子集） |
| `--max-model-len 131072` | 长上下文（math/bbh 长推理链） | `competition_math`/`bbh`/`agieval` |
| `--max-num-seqs 256` | 高并发批处理 | 全部 |
| `--trust-remote-code` | 自定义模型代码 | 全部 |
| `--gpu-memory-utilization 0.92` | 显存利用率 | 全部 |

**tool-call-parser 选择**（vLLM v0.28.0 实测支持列表）：
- **Qwen3 系列 → `qwen3_xml`**（Qwen3 原生 XML 工具格式）或 `qwen3_coder`（代码模型）
- Qwen2.5 工具模型 → `hermes`
- 其他模型按官方文档选（`internlm`/`mistral`/`llama3_json` 等）

> ⚠️ **实测踩坑**：vLLM 未开 `--enable-auto-tool-choice` 时，bfcl_v3 提交即失败：
> `400 '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'`

### 7.3 评测端注意（与 vLLM 配合）

- **humaneval 代码执行**：需 EvalScope 侧 sandbox（`use_sandbox`，Docker 执行 Python），与 vLLM 无关
- **aime24/competition_math 长推理**：27B 单条 46-73s 且打满 max_tokens，建议 limit≤2 或剔除
- **并发批大小**：`--max-num-seqs` 给足即可，EvalScope 侧 `eval_batch_size=8` 默认即可（实测并发提升无额外收益）

### 7.4 GPU 驱动 mismatch 排查（宿主机）

**症状**：`nvidia-smi` 报 `Failed to initialize NVML: Driver/library version mismatch`；起新 GPU 容器报 `nvml error: driver/library version mismatch`。

**原因**：GPU 驱动升级后未重启，内核模块（如 595.71）与用户态库（595.91）版本不一致。

**修复**（需中断 GPU 服务，谨慎操作）：

```bash
# 1. 停所有 GPU 容器 + GPU 进程
docker stop <gpu-container-1> <gpu-container-2>
kill <gpu-process-pid>

# 2. 卸载旧 nvidia 模块（需先无 GPU 引用）
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia

# 3. 重新加载（DKMS 自动用新版本 595.91）
sudo modprobe nvidia && nvidia-smi   # 确认版本一致

# 4. 重建 GPU 容器（7.1 命令）
```

> 重启宿主机同样可解决（更简单但影响面更大）。日常 GPU 驱动升级后应尽快重启。

---

## 八、数据集机制（重要，实测验证）

### 7.1 加载路径

EvalScope 数据集加载（源码实证 `evalscope/api/dataset/hub.py` + 实测）：

```
dataset_hub=MODELSCOPE  -> MsDataset.load(id) 不传 cache_dir，命中 MODELSCOPE_CACHE env 的默认路径
                           <cache>/datasets/<org>___<name>/...（arrow 缓存）
dataset_hub=HUGGINGFACE -> datasets.load_dataset(id)，命中 HF_HOME（镜像内 /data/datasets_cache/hf_home）
dataset_id 是本地路径    -> os.path.exists 命中即本地读取，零网络
```

### 7.2 离线命中的关键（踩坑记录）

1. 构建期预下载必须用 **`MsDataset.load`**（与运行时同调用路径），不能 `dataset_snapshot_download`
2. 预下载**不能传 `cache_dir`**——必须依赖 `MODELSCOPE_CACHE` env，让落盘布局（`<cache>/datasets/<org>___<name>/`）与运行时完全一致
3. 实测：构建期缓存 + 运行时同路径 → 第二次加载 2.3s（命中）；路径不一致 → 永远 miss 联网下载

- **内置**：构建期 `scripts/prefetch_datasets.py`（设置 `MODELSCOPE_CACHE=/data/datasets_cache` 后 `MsDataset.load` 预下载各 split，覆盖 test/validation/val/dev/train）
- **覆盖**：运行时 `-v <host>:/data/datasets_cache` 挂载即整体覆盖内置（宿主数据优先）
- **离线**：纯内网 `HF_HUB_OFFLINE=1` + 内置数据，完全零网络
- **不用 HF 兜底**：HF `snapshot_download` 落盘 `<hf_home>/datasets/<org>__<name>/`，而 EvalScope 运行时查 `MODELSCOPE_CACHE/datasets/<org>___<name>/`（hub.py 不传 cache_dir）——布局不兼容必然 miss，故内置清单只走 MS 通道

### 7.3 预取策略（构建期 `scripts/prefetch_datasets.py`）

按运行时加载语义区分（A800 全量冒烟实证 17/17 通过）：

- **真多 config**（ceval 52 / bbh 27 / agieval 21 / arc 2）：MS 目录 = subset，逐 subset 预取，运行时 `MsDataset.load(subset_name=...)` 命中
- **reformat 数据集**（cmmlu / mmlu_pro / competition_math / bfcl_v3）：MS 单 config，运行时只加载 default 再按数据列（category/level/multi_turn）分组——预取不传 subset_name
- **特殊 config**（gsm8k=`main` / truthful_qa=`multiple_choice` / humaneval=`openai_humaneval`）：显式预取该 config
- **单 config 平铺**（aime24 / gpqa_diamond / hellaswag / winogrande / commonsense_qa / ifeval）：预取 default

自定义预取清单：`--datasets "gsm8k,ceval"`（EvalScope benchmark 名）或 `--datasets-file list.txt`。
