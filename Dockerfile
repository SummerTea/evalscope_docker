# EvalScope 评测服务镜像（内置核心数据集，运行时可挂载覆盖）
#
# 特性（基于源码核实）：
# - PyPI wheel 自带构建好的 React SPA（evalscope.web 包含 dist/**），无需 node 构建阶段
# - evalscope[service] = flask + plotly + perf(aiohttp/uvicorn/numpy)，无 torch
# - 入口 `evalscope service` 起 Flask Web 服务（任务/报告/对比/Arena），默认端口 9000
# - 评测走 OpenAI 兼容 API（EVALSCOPE_BASE_URL 指向内网 vLLM/SGLang），纯 CPU 可跑
# - 内置核心数据集到 /data/datasets_cache（EvalScope 按 MODELSCOPE_CACHE/HF_HOME 命中，零网络）
# - 运行时可挂载 -v <host>:/data/datasets_cache 覆盖内置数据集
#
# 构建参数：
#   EVALSCOPE_VERSION    EvalScope 版本（默认 1.11.1）
#   PIP_INDEX_URL        pip 源（内网可换 https://mirrors.huaweicloud.com/repository/pypi/simple）
#   PREFETCH_DATASETS    是否预取核心数据集（true/false，默认 true）
#   PREFETCH_CHANNEL     预取通道优先级（hf-first 适配 GitHub Actions / ms-first 适配内网）
#   HF_ENDPOINT_BUILD    构建期 HF 端点（默认官方 https://huggingface.co）
#   DATASETS_EXTRA       追加预取数据集（逗号分隔，追加到默认核心集）
#
# 注意（GitHub Actions runner 磁盘 ~14GB）：内置数据仅限精选核心集 ~1GB；
# 全量数据（20-40GB+）请运行时挂载覆盖。

ARG EVALSCOPE_VERSION=1.11.1
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PREFETCH_DATASETS=true
# MS-first：适配 EvalScope 默认 ModelScope 数据源（dataset_id 即 MS ID）
ARG PREFETCH_CHANNEL=ms-first
ARG HF_ENDPOINT_BUILD=https://huggingface.co
ARG DATASETS_EXTRA=

# ---------- stage 1: 数据集预取层 ----------
FROM python:3.12-slim AS datasets-prefetch

ARG PREFETCH_DATASETS
ARG PREFETCH_CHANNEL
ARG HF_ENDPOINT_BUILD
ARG DATASETS_EXTRA
ARG PIP_INDEX_URL

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/datasets_cache/hf_home \
    # uv 使用（Rust 实现，pip 10-50x）
    UV_NO_CACHE=1 \
    UV_INDEX_URL=${PIP_INDEX_URL}

# 用 uv 替代 pip：并行下载 + 极速解析（GitHub Actions 构建 pip 慢的根治方案）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY scripts/prefetch_datasets.py /opt/prefetch_datasets.py

RUN uv pip install --system --python /usr/local/bin/python \
        "huggingface_hub" "modelscope[datasets]"

RUN if [ "${PREFETCH_DATASETS}" = "true" ]; then \
        HF_ENDPOINT="${HF_ENDPOINT_BUILD}" MODELSCOPE_CACHE=/data/datasets_cache \
            python /opt/prefetch_datasets.py \
            --output /data/datasets_cache \
            ${DATASETS_EXTRA:+--datasets "${DATASETS_EXTRA}"}; \
    fi

# ---------- stage 2: 运行层 ----------
FROM python:3.12-slim

ARG EVALSCOPE_VERSION
ARG PIP_INDEX_URL

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_INDEX_URL=${PIP_INDEX_URL} \
    # 评测结果与数据集缓存统一放 /data（运行时挂载卷）
    EVALSCOPE_OUTPUTS_DIR=/data/outputs \
    HF_HOME=/data/datasets_cache/hf_home \
    MODELSCOPE_CACHE=/data/datasets_cache \
    EVALSCOPE_BASE_URL="" \
    EVALSCOPE_API_KEY="EMPTY"

WORKDIR /app

# 用 uv 安装 EvalScope（含 web service 前端产物）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --system --python /usr/local/bin/python \
    "evalscope[service]==${EVALSCOPE_VERSION}"

# 安装 τ²-bench（agent 评测：airline/retail/telecom 客服域）
# 与 EvalScope tau2_bench adapter 要求一致（sierra-research/tau2-bench@v0.2.0）
# 注意：tau2 与 tau3 不能同环境共存（同一 PyPI 包名不同版本），本镜像按 tau2 接入
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && uv pip install --system --python /usr/local/bin/python \
    "tau2 @ git+https://github.com/sierra-research/tau2-bench@v0.2.0" \
    && uv pip install --system --python /usr/local/bin/python \
    "litellm" "fastapi" "pandas"

# 验证前端产物与 CLI 就绪
RUN python -c "import os; d=os.path.join(os.path.dirname(os.path.dirname(__import__('evalscope').__file__)),'evalscope','web','dist'); assert os.path.isdir(d), f'missing web dist: {d}'; print('web dist OK:', d)" \
    && evalscope --help > /dev/null

# 内置数据集缓存（构建期已灌入；运行时挂载覆盖）
COPY --from=datasets-prefetch /data/datasets_cache /data/datasets_cache

# 数据卷：评测输出、数据集缓存（挂载覆盖内置）
VOLUME ["/data/outputs", "/data/datasets_cache"]

EXPOSE 9000

ENTRYPOINT ["evalscope", "service", "--host", "0.0.0.0", "--port", "9000", "--outputs", "/data/outputs"]
