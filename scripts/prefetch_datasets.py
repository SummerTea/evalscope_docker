#!/usr/bin/env python3
"""构建期核心数据集预取脚本（Docker 构建时运行）。

目标：让 EvalScope 运行时**零网络**加载核心数据集（内网离线可用）。
机制（evalscope/api/dataset/hub.py + meta.py 实证）：
  - 每个 benchmark 的 `dataset_id` 是 ModelScope ID（如 AI-ModelScope/gsm8k）
  - dataset_hub=MODELSCOPE -> MsDataset.load(id)，命中 MODELSCOPE_CACHE 缓存
  所以构建期把数据集灌进 MS 缓存目录，运行时 `MODELSCOPE_CACHE` 命中即零网络。

通道策略（本脚本为 MS-only，适配 EvalScope 默认数据源）：
  - ModelScope（EvalScope 默认通道，dataset_id 即 MS ID）——唯一通道
  - 不用 HF 兜底：HF snapshot_download 落盘 HF_HOME 布局，而 EvalScope 运行时
    走 MsDataset.load 查 MODELSCOPE_CACHE 布局，两者不兼容，兜底数据必然 miss 联网重下。

用法：
  python prefetch_datasets.py --output /data/datasets_cache
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
    'truthful_qa': ('evalscope/truthful_qa', 'TruthfulQA 事实'),
    'commonsense_qa': ('extraordinarylab/commonsense-qa', 'CommonsenseQA'),
    # 指令遵循
    'ifeval': ('opencompass/ifeval', 'IFEval 指令遵循'),
    # 代码
    'humaneval': ('opencompass/humaneval', 'HumanEval 代码'),
    # 函数调用
    'bfcl_v3': ('AI-ModelScope/bfcl_v3', 'BFCL-v3 函数调用（17 子集）'),
}

# MS 无时的 HF 兜底映射（EvalScope dataset_id -> HF repo）
# 注意：已弃用——HF 兜底落盘 HF_HOME 布局，与 EvalScope 运行时 MODELSCOPE_CACHE 布局不兼容，
# 必然 miss 联网重下（详见 hf_download 弃用说明）。保留映射仅作参考，不再使用。
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

    失败策略（实测踩坑）：
    - split 不存在（如只有 validation 的 truthful_qa 试 test）：静默跳过（loaded_any 不受影响）
    - split 存在但下载失败（CDN 瞬时故障，如 ceval 的 business_administration/dev 曾报
      Network is unreachable）：**重试 3 次**；仍失败则视为该数据集不完整，
      loaded_any 记 False → 构建 FAIL（宁可构建失败暴露，也不内置残缺缓存导致运行时联网）。
    """
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        print(f'  [warn] modelscope 未安装，跳过 MS 通道: {dataset_id}', flush=True)
        return False
    try:
        # 预下载各 split（test 为主；validation 供 eval_split=validation 的 benchmark 用，
        # val/dev/train 供 fewshot 用，缺失的自动跳过）
        splits = ['test', 'validation', 'val', 'dev', 'train']
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
                # split 存在但下载失败：重试 3 次（CDN 瞬时故障可恢复），仍失败则数据集不完整
                retried = False
                for attempt in range(1, 4):
                    print(f'  [warn] MS 预缓存 {dataset_id} split={split} 失败(重试 {attempt}/3): '
                          f'{type(e).__name__}: {str(e)[:120]}', flush=True)
                    try:
                        MsDataset.load(dataset_name=dataset_id, split=split)
                        loaded_any = True
                        retried = True
                        print(f'  [ok] MS 预缓存 {dataset_id} split={split} (重试成功)', flush=True)
                        break
                    except Exception as e2:
                        e = e2
                if retried:
                    continue
                print(f'  [warn] MS 预缓存 {dataset_id} split={split} 重试后仍失败: '
                      f'{type(e).__name__}: {str(e)[:120]}', flush=True)
                return False  # 数据集不完整，构建 FAIL（不内置残缺缓存）
        return loaded_any
    except Exception as e:
        print(f'  [warn] ModelScope 预下载失败 {dataset_id}: {type(e).__name__}: {str(e)[:120]}', flush=True)
        return False


def hf_download(hf_id: str, hf_home: str, endpoint: str) -> bool:
    """【已弃用】HF 缓存下载（HF_HOME 布局）。

    弃用原因（实测）：HF snapshot_download 落盘 <hf_home>/datasets/<org>__<name>/...，
    而 EvalScope 运行时 load_dataset_from_hub（hub.py:89-104）走 MsDataset.load 查
    MODELSCOPE_CACHE/datasets/<org>___<name>/...（不传 cache_dir）——两者布局不兼容，
    兜底数据运行时必然 miss 联网重下，等于没内置。故主流程只走 MS 通道。
    此外同环境 modelscope[datasets] 会遮蔽 huggingface_hub 顶层导出，导致
    snapshot_download 被劫持报 NotExistError/E3020（URL 指向 modelscope.cn）。

    保留此函数仅为向后兼容（--datasets 显式指定 MS 无的 repo 时仍可手动兜底），
    内置清单 CORE_BENCHMARKS 不再走该通道。
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
                        help='通道优先级（默认 ms-first：内置清单走 MS 通道，HF 兜底已弃用）')
    parser.add_argument('--datasets', default='', help='显式 benchmark 清单（逗号分隔，覆盖默认）')
    parser.add_argument('--datasets-file', default='', help='benchmark 清单文件（每行一个，# 注释）')
    parser.add_argument('--hf-endpoint', default='https://huggingface.co', help='HF 端点（仅 hf-first 手动兜底时使用）')
    args = parser.parse_args()

    if args.datasets_file:
        with open(args.datasets_file) as f:
            benchmarks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    elif args.datasets:
        benchmarks = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        benchmarks = list(CORE_BENCHMARKS.keys())

    os.makedirs(args.output, exist_ok=True)
    # 全局 env 一次性设置（EvalScope 运行时同路径命中）
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
        """下载单个 benchmark，返回 (benchmark, ok)。

        ms-first：只走 MS 通道（EvalScope 默认数据源，落盘 MODELSCOPE_CACHE 布局与运行时一致）；
        HF 兜底已弃用（布局不兼容必然 miss），MS 失败即记 FAIL。
        """
        ms_id, hf_id = resolve_dataset_id(benchmark)
        print(f'===== {benchmark} (MS:{ms_id}) =====', flush=True)
        if args.channel == 'hf-first':
            # 仅显式指定 hf-first 时允许 HF 兜底（手动场景，内置清单不用）
            done = try_hf(hf_id) or try_ms(ms_id)
        else:
            done = try_ms(ms_id)
        return benchmark, done

    # 串行下载（modelscope MsDataset 库级全局 monkey-patch 并发不安全，实测并发会导致
    # 个别数据集失败；串行稳定且构建时长可接受）
    results = [download_one(b) for b in benchmarks]

    ok = [b for b, done in results if done]
    failed = [b for b, done in results if not done]

    print(f'\n=== 完成: OK={len(ok)} FAIL={len(failed)} ===')
    for f in failed:
        print(f'  FAIL: {f}', flush=True)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
