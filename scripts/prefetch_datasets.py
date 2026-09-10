#!/usr/bin/env python3
"""构建期核心数据集预取脚本（Docker 构建时运行）。

目标：让 EvalScope 运行时**零网络**加载核心数据集（内网离线可用）。
机制（evalscope/api/dataset/hub.py + meta.py 实证）：
  - 每个 benchmark 的 `dataset_id` 是 ModelScope ID（如 AI-ModelScope/gsm8k）
  - dataset_hub=MODELSCOPE -> MsDataset.load(id)，命中 MODELSCOPE_CACHE 缓存
  所以构建期把数据集灌进 MS 缓存目录，运行时 `MODELSCOPE_CACHE` 命中即零网络。

通道策略（本脚本为 MS-first，适配 EvalScope 默认数据源）：
  1. ModelScope（EvalScope 默认通道，dataset_id 即 MS ID）——主通道
  2. HuggingFace 兜底（仅当 MS 无该 ID 时，走 HF 缓存 HF_HOME）

用法：
  python prefetch_datasets.py --output /data/datasets_cache
  python prefetch_datasets.py --channel ms-first            # 默认（适配 EvalScope）
  python prefetch_datasets.py --datasets "gsm8k,ceval"      # 显式清单（EvalScope benchmark 名）
  python prefetch_datasets.py --datasets-file datasets.txt  # 清单文件（每行一个 benchmark 名）
"""

import argparse
import os
import sys

# 内置测试集：EvalScope benchmark 名 -> (MS dataset_id, 说明)
# 覆盖维度：数学/中文/知识/推理/常识/指令/代码/综合/agent/函数调用（除多模态外全维度）
# 体积预算 ~1-2GB（GitHub Actions runner 磁盘 ~14GB 内可控）
CORE_BENCHMARKS = {
    # 数学推理
    'gsm8k': ('AI-ModelScope/gsm8k', 'GSM8K 数学推理'),
    'aime24': ('evalscope/aime24', 'AIME-2024 竞赛数学'),
    'competition_math': ('evalscope/competition_math', 'MATH 竞赛'),
    # 中文知识
    'ceval': ('evalscope/ceval', 'C-Eval 中文（52 子集）'),
    'cmmlu': ('evalscope/cmmlu', 'CMMLU 中文（67 子集）'),
    # 通用知识
    'mmlu_pro': ('TIGER-Lab/MMLU-Pro', 'MMLU-Pro 知识（14 子集）'),
    'gpqa_diamond': ('AI-ModelScope/gpqa_diamond', 'GPQA-Diamond 科学'),
    # 推理
    'bbh': ('evalscope/bbh', 'BBH 推理（27 子集）'),
    'arc': ('allenai/ai2_arc', 'ARC 推理'),
    'agieval': ('opencompass/agieval', 'AGIEval 综合（21 子集中英）'),
    # 常识
    'hellaswag': ('evalscope/hellaswag', 'HellaSwag 常识'),
    'winogrande': ('AI-ModelScope/winogrande_val', 'Winogrande 常识'),
    'truthful_qa': ('evalscope/truthful_qa', 'TruthfulQA 事实'),
    'commonsense_qa': ('extraordinarylab/commonsense-qa', 'CommonsenseQA'),
    # 指令遵循
    'ifeval': ('opencompass/ifeval', 'IFEval 指令遵循'),
    # 代码
    'humaneval': ('opencompass/humaneval', 'HumanEval 代码'),
    # Agent
    'tau2_bench': ('evalscope/tau2-bench-data', 'τ²-bench 客服 agent（airline/retail/telecom）'),
    # 函数调用
    'bfcl_v3': ('AI-ModelScope/bfcl_v3', 'BFCL-v3 函数调用（17 子集）'),
}

# MS 无时的 HF 兜底映射（EvalScope dataset_id -> HF repo）
MS_TO_HF = {
    'AI-ModelScope/gsm8k': 'openai/gsm8k',
    'evalscope/ceval': 'ceval/ceval-exam',
    'evalscope/cmmlu': 'haonan-li/cmmlu',
    'opencompass/humaneval': 'openai/openai_humaneval',
    'TIGER-Lab/MMLU-Pro': 'TIGER-Lab/MMLU-Pro',
    'allenai/ai2_arc': 'allenai/ai2_arc',
    'evalscope/bbh': 'lukaemon/bbh',
    'AI-ModelScope/gpqa_diamond': 'Idavidrein/gpqa',
    'evalscope/truthful_qa': 'truthfulqa/truthful_qa',
    'opencompass/ifeval': 'google/IFEval',
    'AI-ModelScope/winogrande_val': 'allenai/winogrande',
    'extraordinarylab/commonsense-qa': 'tau/commonsense_qa',
    'evalscope/hellaswag': 'Rowan/hellaswag',
}


def ms_download(dataset_id: str, cache_dir: str) -> bool:
    """ModelScope 缓存下载；命中缓存则跳过。"""
    try:
        from modelscope import dataset_snapshot_download
    except ImportError:
        print(f'  [warn] modelscope 未安装，跳过 MS 通道: {dataset_id}', flush=True)
        return False
    try:
        # modelscope 缓存布局: <cache>/hub/datasets/<org>/<name>
        org, _, name = dataset_id.partition('/')
        hit = os.path.isdir(os.path.join(cache_dir, 'hub', 'datasets', org, name))
        if hit:
            print(f'  [skip] MS 缓存命中: {dataset_id}', flush=True)
            return True
        dataset_snapshot_download(dataset_id, cache_dir=cache_dir)
        print(f'  [ok] ModelScope: {dataset_id}', flush=True)
        return True
    except Exception as e:
        print(f'  [warn] ModelScope 下载失败 {dataset_id}: {type(e).__name__}: {e}', flush=True)
        return False


def hf_download(hf_id: str, hf_home: str, endpoint: str) -> bool:
    """HF 缓存下载（HF_HOME 布局），命中缓存则跳过。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(f'  [warn] huggingface_hub 未安装，跳过 HF 通道: {hf_id}', flush=True)
        return False
    try:
        os.environ['HF_ENDPOINT'] = endpoint
        os.environ['HF_HOME'] = hf_home
        snapshot_download(repo_id=hf_id, repo_type='dataset')
        print(f'  [ok] HF: {hf_id} (endpoint={endpoint})', flush=True)
        return True
    except Exception as e:
        print(f'  [warn] HF 下载失败 {hf_id}: {type(e).__name__}: {e}', flush=True)
        return False


def resolve_dataset_id(benchmark: str) -> tuple[str, str]:
    """benchmark 名 -> (MS dataset_id, HF 兜底 ID)。"""
    entry = CORE_BENCHMARKS.get(benchmark)
    if entry:
        return entry[0], MS_TO_HF.get(entry[0], entry[0])
    return benchmark, benchmark


def main():
    parser = argparse.ArgumentParser(description='预取核心评测数据集到缓存目录')
    parser.add_argument('--output', default='/data/datasets_cache', help='缓存根目录（默认 /data/datasets_cache）')
    parser.add_argument('--channel', default='ms-first', choices=['ms-first', 'hf-first'],
                        help='通道优先级（默认 ms-first，适配 EvalScope 默认 MS 数据源）')
    parser.add_argument('--datasets', default='', help='显式 benchmark 清单（逗号分隔，覆盖默认）')
    parser.add_argument('--datasets-file', default='', help='benchmark 清单文件（每行一个，# 注释）')
    parser.add_argument('--hf-endpoint', default='https://huggingface.co', help='HF 端点（内网可换 https://hf-mirror.com）')
    args = parser.parse_args()

    if args.datasets_file:
        with open(args.datasets_file) as f:
            benchmarks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    elif args.datasets:
        benchmarks = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        benchmarks = list(CORE_BENCHMARKS.keys())

    os.makedirs(args.output, exist_ok=True)
    hf_home = os.path.join(args.output, 'hf_home')
    os.makedirs(hf_home, exist_ok=True)

    def try_ms(dataset_id):
        return ms_download(dataset_id, args.output)

    def try_hf(hf_id):
        return hf_download(hf_id, hf_home, args.hf_endpoint)

    ok, failed = [], []
    for benchmark in benchmarks:
        ms_id, hf_id = resolve_dataset_id(benchmark)
        print(f'===== {benchmark} (MS:{ms_id}) =====', flush=True)
        if args.channel == 'ms-first':
            done = try_ms(ms_id) or try_hf(hf_id)
        else:
            done = try_hf(hf_id) or try_ms(ms_id)
        (ok if done else failed).append(benchmark)

    print(f'\n=== 完成: OK={len(ok)} FAIL={len(failed)} ===')
    for f in failed:
        print(f'  FAIL: {f}', flush=True)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
