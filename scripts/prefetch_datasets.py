#!/usr/bin/env python3
"""构建期核心数据集预取脚本（Docker 构建时运行）。

目标：让 EvalScope 运行时**零网络**加载核心数据集。
机制（evalscope/api/dataset/hub.py 实证）：
  - dataset_hub=MODELSCOPE  ->  MsDataset.load(id)，命中 MODELSCOPE_CACHE 缓存
  - dataset_hub=HUGGINGFACE ->  datasets.load_dataset(id)，命中 HF_HOME 缓存
所以构建期把数据集灌进对应缓存目录，运行时设同路径环境变量即命中。

通道优先级（按构建环境反转，勿写死）：
  - GitHub Actions（公网）：HF 官方直连最顺（无 AWS CDN/xethub 限制）-> ModelScope 兜底
  - 内网/自建 runner（如 A800）：ModelScope 直连稳定 -> HF(hf-mirror) 兜底
  用 --channel hf-first | ms-first 控制（默认 hf-first，适配 GitHub Actions）。

用法：
  python prefetch_datasets.py --output /data/datasets_cache
  python prefetch_datasets.py --channel hf-first            # Actions 默认
  python prefetch_datasets.py --channel ms-first --hf-endpoint https://hf-mirror.com
  python prefetch_datasets.py --datasets "cais/mmlu,openai/gsm8k"
  python prefetch_datasets.py --datasets-file datasets.txt
"""

import argparse
import os
import sys


# 默认核心集（对应 EvalScope 常用 benchmark 的 dataset_id）
# 体积预算 ~600MB-1GB，适配 GitHub Actions runner 磁盘（~14GB）
CORE_DATASETS = [
    # 冒烟层（tinyBenchmarks 精选子集，极小）
    'tinyBenchmarks/tinyMMLU',
    'tinyBenchmarks/tinyGSM8k',
    'tinyBenchmarks/tinyAI2_arc',
    'tinyBenchmarks/tinyHellaswag',
    'tinyBenchmarks/tinyTruthfulQA',
    'tinyBenchmarks/tinyWinogrande',
    # 常用客观基准（标准验收层）
    'cais/mmlu',            # MMLU 知识多选（~160MB）
    'openai/gsm8k',         # GSM8K 数学推理
    'ceval/ceval-exam',     # C-Eval 中文
    'haonan-li/cmmlu',      # CMMLU 中文
    'openai/openai_humaneval',  # HumanEval 代码
    'EleutherAI/wikitext_document_level',  # wikitext（PPL 语言建模）
]

# ModelScope ID -> HuggingFace ID（名称不同时的映射）
MS_TO_HF = {
    'ceval/ceval-exam': 'ceval/ceval-exam',
    'haonan-li/cmmlu': 'haonan-li/cmmlu',
}


def ms_download(dataset_id: str, cache_dir: str) -> bool:
    """ModelScope 缓存下载；命中缓存则跳过。"""
    try:
        from modelscope import dataset_snapshot_download
    except ImportError:
        print(f'  [warn] modelscope 未安装，跳过 MS 通道: {dataset_id}', flush=True)
        return False
    try:
        # modelscope 缓存布局: <cache>/hub/datasets/<org>/<name>；已存在则命中跳过
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
        # HF 缓存布局: <hf_home>/hub/datasets--org--name；snapshot_download 幂等命中
        snapshot_download(repo_id=hf_id, repo_type='dataset')
        print(f'  [ok] HF: {hf_id} (endpoint={endpoint})', flush=True)
        return True
    except Exception as e:
        print(f'  [warn] HF 下载失败 {hf_id}: {type(e).__name__}: {e}', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='预取核心评测数据集到缓存目录')
    parser.add_argument('--output', default='/data/datasets_cache', help='缓存根目录（默认 /data/datasets_cache）')
    parser.add_argument('--channel', default='hf-first', choices=['hf-first', 'ms-first'],
                        help='通道优先级（默认 hf-first，适配 GitHub Actions；内网构建用 ms-first）')
    parser.add_argument('--datasets', default='', help='显式数据集清单（逗号分隔，覆盖默认）')
    parser.add_argument('--datasets-file', default='', help='数据集清单文件（每行一个 ID，# 注释）')
    parser.add_argument('--hf-endpoint', default='https://huggingface.co', help='HF 端点（内网可换 https://hf-mirror.com）')
    args = parser.parse_args()

    if args.datasets_file:
        with open(args.datasets_file) as f:
            datasets = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        datasets = list(CORE_DATASETS)

    os.makedirs(args.output, exist_ok=True)
    hf_home = os.path.join(args.output, 'hf_home')
    os.makedirs(hf_home, exist_ok=True)

    def try_primary(dataset_id):
        """按 channel 决定先试哪个通道。"""
        if args.channel == 'hf-first':
            hf_id = MS_TO_HF.get(dataset_id, dataset_id)
            return hf_download(hf_id, hf_home, args.hf_endpoint)
        return ms_download(dataset_id, args.output)

    def try_fallback(dataset_id):
        if args.channel == 'hf-first':
            return ms_download(dataset_id, args.output)
        hf_id = MS_TO_HF.get(dataset_id, dataset_id)
        return hf_download(hf_id, hf_home, args.hf_endpoint)

    ok, failed = [], []
    for dataset_id in datasets:
        print(f'===== {dataset_id} =====', flush=True)
        if try_primary(dataset_id):
            ok.append(dataset_id)
        elif try_fallback(dataset_id):
            ok.append(dataset_id)
        else:
            failed.append(dataset_id)

    print(f'\n=== 完成: OK={len(ok)} FAIL={len(failed)} ===')
    for f in failed:
        print(f'  FAIL: {f}', flush=True)
    # fail loudly：构建期缺数据应显式失败（不需要预取的场景用户可关掉预取层）
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
