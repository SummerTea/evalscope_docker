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
# 覆盖维度：数学/中文/知识/推理/常识/指令/代码/函数调用（除多模态外全维度）
# 体积预算 ~1-2GB（GitHub Actions runner 磁盘 ~14GB 内可控）
# 注意：tau2_bench 已排除——其数据加载走独立的 resolve_snapshot_or_local_path 快照路径
# （非 MsDataset.load 主流程），且数据格式特殊（DatasetGenerationError 实测），
# 运行时按需拉取或挂载覆盖即可，不影响内置离线集。
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
    'commonsense_qa': ('extraordinarylab/commonsense-qa', 'CommonsenseQA'),
    # 指令遵循
    'ifeval': ('opencompass/ifeval', 'IFEval 指令遵循'),
    # 代码
    'humaneval': ('opencompass/humaneval', 'HumanEval 代码'),
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
    """ModelScope 预下载并缓存数据集（与 EvalScope 运行时 MsDataset.load 同路径，命中即离线）。

    关键（实测验证）：
    - EvalScope 的 load_dataset_from_hub 调用 MsDataset.load(dataset_name=id)，不传 cache_dir，
      走 MODELSCOPE_CACHE 环境变量默认路径 <cache>/datasets/<org>___<name>/...
    - 因此构建期必须**不传 cache_dir**（依赖 MODELSCOPE_CACHE env），让落盘布局与运行时一致，
      否则构建期 cache_dir=X 落的 <X>/<name>/... 与运行时 <X>/datasets/<name>/... 不一致，永远 miss。
    - 本函数依赖调用方已设 MODELSCOPE_CACHE=cache_dir（main 中设置）。
    """
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        print(f'  [warn] modelscope 未安装，跳过 MS 通道: {dataset_id}', flush=True)
        return False
    try:
        # 预下载各 split（test 为主；val/dev/train 供 fewshot 用，缺失的自动跳过）
        splits = ['test', 'val', 'dev', 'train']
        loaded_any = False
        for split in splits:
            try:
                MsDataset.load(dataset_name=dataset_id, split=split)  # 不传 cache_dir，走 MODELSCOPE_CACHE
                loaded_any = True
                print(f'  [ok] MS 预缓存 {dataset_id} split={split}', flush=True)
            except Exception as e:
                msg = str(e)[:120].lower()
                if 'split' in msg or 'keyerror' in type(e).__name__.lower() or 'not found' in msg:
                    continue  # 该 split 不存在，跳过
                print(f'  [warn] MS 预缓存 {dataset_id} split={split} 失败: {type(e).__name__}: {str(e)[:120]}', flush=True)
        return loaded_any
    except Exception as e:
        print(f'  [warn] ModelScope 预下载失败 {dataset_id}: {type(e).__name__}: {str(e)[:120]}', flush=True)
        return False


def hf_download(hf_id: str, hf_home: str, endpoint: str) -> bool:
    """HF 缓存下载（HF_HOME 布局），命中缓存则跳过。

    注意：HF_ENDPOINT/HF_HOME 由 main 一次性设置（全局 env），本函数不再修改。
    注意：必须显式 `import huggingface_hub` 后取 snapshot_download——
    同环境装了 modelscope[datasets] 会遮蔽 huggingface_hub 的顶层导出，
    直接 `from huggingface_hub import snapshot_download` 可能拿到 modelscope 实现
    （实测报 NotExistError/E3020 且 URL 指向 modelscope.cn）。
    """
    try:
        import huggingface_hub
    except ImportError:
        print(f'  [warn] huggingface_hub 未安装，跳过 HF 通道: {hf_id}', flush=True)
        return False
    try:
        huggingface_hub.snapshot_download(repo_id=hf_id, repo_type='dataset')
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
    parser.add_argument('--parallel', type=int, default=4, help='并发下载数（默认 4，数据集间无依赖可并行）')
    args = parser.parse_args()

    if args.datasets_file:
        with open(args.datasets_file) as f:
            benchmarks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    elif args.datasets:
        benchmarks = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        benchmarks = list(CORE_BENCHMARKS.keys())

    os.makedirs(args.output, exist_ok=True)
    # 全局 env 一次性设置（并发下载时多线程改 env 会互相覆盖，严禁在下载函数内修改）
    os.environ['MODELSCOPE_CACHE'] = args.output
    os.environ['HF_ENDPOINT'] = args.hf_endpoint
    hf_home = os.path.join(args.output, 'hf_home')
    os.environ['HF_HOME'] = hf_home
    os.makedirs(hf_home, exist_ok=True)

    def try_ms(dataset_id):
        return ms_download(dataset_id, args.output)

    def try_hf(hf_id):
        return hf_download(hf_id, hf_home, args.hf_endpoint)

    def download_one(benchmark):
        """下载单个 benchmark（MS 优先或 HF 优先），返回 (benchmark, ok)。"""
        ms_id, hf_id = resolve_dataset_id(benchmark)
        print(f'===== {benchmark} (MS:{ms_id}) =====', flush=True)
        if args.channel == 'ms-first':
            done = try_ms(ms_id) or try_hf(hf_id)
        else:
            done = try_hf(hf_id) or try_ms(ms_id)
        return benchmark, done

    # 并发下载（数据集间无依赖；Modelscope 库线程安全即可并行）
    import concurrent.futures
    max_workers = max(1, min(args.parallel, len(benchmarks)))
    results = []
    if len(benchmarks) == 1 or args.parallel <= 1:
        for b in benchmarks:
            results.append(download_one(b))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(download_one, b): b for b in benchmarks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append((futures[fut], False))
                    print(f'  [warn] {futures[fut]} 并发任务异常: {e}', flush=True)

    # 失败项串行重试一次（modelscope MsDataset 并发不安全，串行可规避）
    retry_failed = [b for b, done in results if not done]
    if retry_failed:
        print(f'\n=== 并发完成，{len(retry_failed)} 项失败，串行重试 ===', flush=True)
        for b in retry_failed:
            results = [(bb, dd) for bb, dd in results if bb != b]
            results.append(download_one(b))

    ok = [b for b, done in results if done]
    failed = [b for b, done in results if not done]

    print(f'\n=== 完成: OK={len(ok)} FAIL={len(failed)} ===')
    for f in failed:
        print(f'  FAIL: {f}', flush=True)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
