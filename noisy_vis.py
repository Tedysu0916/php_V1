import os
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from datasets.bases import tokenize
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
import tqdm
import textwrap

from utils.simple_tokenizer import SimpleTokenizer


def encode_custom_text(model, text, tokenizer, device):
    model.eval()
    with torch.no_grad():
        tokens = tokenize(text, tokenizer=SimpleTokenizer(), text_length=77, truncate=True).unsqueeze(0).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features


def get_retrieval_results(custom_text, model, gfeats, gids, img_paths, topk=10, device='cuda'):
    """获取检索结果，不保存图片"""
    text_feat = encode_custom_text(model, custom_text, None, device)
    similarity = text_feat @ gfeats.t()
    scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=True)

    indices = indices.cpu().numpy()[0]
    scores = scores.cpu().numpy()[0]
    image_paths = [img_paths[i] for i in indices]
    image_ids = gids[indices].cpu().numpy()

    return {
        'query': custom_text,
        'indices': indices,
        'scores': scores,
        'image_ids': image_ids,
        'image_paths': image_paths
    }


def plot_comparison_figure(baseline_result, variant_results, variant_names,
                           output_path, topk=10, img_size=(128, 256)):
    """
    将baseline和所有变种的检索结果垂直拼接成一张对比图

    Args:
        baseline_result: baseline的检索结果字典
        variant_results: 各变种的检索结果字典列表
        variant_names: 变种名称列表
        output_path: 输出图片路径
        topk: 显示top-k个结果
        img_size: 单张图片大小 (width, height)
    """
    n_rows = 1 + len(variant_results)  # baseline + 各变种
    n_cols = topk

    # 计算图片尺寸
    img_w, img_h = img_size
    fig_width = (n_cols + 2) * 1.5  # 额外留出左侧标签空间
    fig_height = n_rows * 3.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # 确保axes是2D数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    all_results = [baseline_result] + variant_results
    all_names = ['Baseline'] + variant_names

    for row_idx, (result, name) in enumerate(zip(all_results, all_names)):
        for col_idx in range(topk):
            ax = axes[row_idx, col_idx]

            # 加载并显示图片
            img = Image.open(result['image_paths'][col_idx])
            img = img.resize(img_size)
            ax.imshow(img)

            # 设置边框颜色：检查是否与baseline的top1匹配
            if result['indices'][col_idx] == baseline_result['indices'][0]:
                for spine in ax.spines.values():
                    spine.set_color('steelblue')
                    spine.set_linewidth(3)
                rank_label = f"Rank {col_idx + 1} ✓"
            else:
                for spine in ax.spines.values():
                    spine.set_color('steelblue')
                    spine.set_linewidth(1.5)
                rank_label = f"Rank {col_idx + 1}"

            # 显示分数
            ax.set_title(f"{rank_label}\n({result['scores'][col_idx]:.3f})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # 在最左侧添加行标签（变种类型名称）
        axes[row_idx, 0].set_ylabel(name, fontsize=10, fontweight='bold', rotation=0,
                                    labelpad=60, ha='right', va='center')

    # 添加总标题：显示baseline query（自动换行）
    wrapped_query = "\n".join(textwrap.wrap(baseline_result['query'], width=120))
    fig.suptitle(f"Query: {wrapped_query}", fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.08, 0.02, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"对比图已保存: {output_path}")


def run_robustness_comparison(model, gfeats, gids, img_paths, output_base_dir, device='cuda'):
    """运行鲁棒性对比测试"""

    if not op.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 定义两组baseline及其变种
    test_cases = [
        {
            "name": "sample",
            "baseline": "a man wearing a black coat and blue jeans walking on the street.",
            "variants": {
                "Synonym": "a guy in a dark tee and denim pants strolling in the building.",
                "Attr Emphasis": "a man in blue jeans and a black coat.",
                "Color Var": "a man wearing a navy blue coat and dark jeans walking on the road.",
                # "Partial": "A woman in a light-colored dress carrying a plant and looking down.",
                # "Word Shuffle": "Carrying a plant and looking down, the woman is dressed in dark shoes and a light-colored dress."
            }
        },
        # {
        #     "name": "woman_with_purse",
        #     "baseline": "The woman is wearing knee-length blue shorts and a green and grey striped shirt. She is holding a white purse in her left hand.",
        #     "variants": {
        #         "Synonym": "The lady is dressed in knee-length blue shorts and a green and grey striped top. She is grasping a white handbag in her left hand.",
        #         "Attr Emphasis": "A woman holding a white purse in her left hand is wearing knee-length blue shorts and a green and grey striped shirt.",
        #         "Color Var": "The woman is wearing knee-length navy shorts and a green and silver striped shirt. She is holding an ivory purse in her left hand.",
        #         # "Partial": "A woman in blue shorts and a green-grey striped shirt holding a white purse.",
        #         # "Word Shuffle": "Holding a white purse in her left hand, the woman is dressed in a green and grey striped shirt with knee-length blue shorts."
        #     }
        # }
    ]

    import pandas as pd
    from scipy.stats import spearmanr

    def compute_overlap_at_k(indices1, indices2, k):
        set1 = set(indices1[:k])
        set2 = set(indices2[:k])
        return len(set1 & set2) / k

    def compute_rank_correlation(indices1, indices2, k=10):
        common = set(indices1[:k]) & set(indices2[:k])
        if len(common) < 2:
            return 0.0
        ranks1 = {idx: rank for rank, idx in enumerate(indices1[:k])}
        ranks2 = {idx: rank for rank, idx in enumerate(indices2[:k])}
        r1 = [ranks1[idx] for idx in common]
        r2 = [ranks2[idx] for idx in common]
        corr, _ = spearmanr(r1, r2)
        return corr if not np.isnan(corr) else 0.0

    all_metrics = []

    for case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Processing: {case['name']}")
        print(f"{'=' * 60}")

        # 获取baseline结果
        baseline_result = get_retrieval_results(
            case['baseline'], model, gfeats, gids, img_paths, topk=10, device=device
        )

        # 获取所有变种结果
        variant_results = []
        variant_names = []

        for var_name, var_query in case['variants'].items():
            var_result = get_retrieval_results(
                var_query, model, gfeats, gids, img_paths, topk=10, device=device
            )
            variant_results.append(var_result)
            variant_names.append(var_name)

            # 计算指标
            metrics = {
                'case': case['name'],
                'variant': var_name,
                'overlap@1': compute_overlap_at_k(baseline_result['indices'], var_result['indices'], 1) * 100,
                'overlap@5': compute_overlap_at_k(baseline_result['indices'], var_result['indices'], 5) * 100,
                'overlap@10': compute_overlap_at_k(baseline_result['indices'], var_result['indices'], 10) * 100,
                'score_retention': (var_result['scores'][0] / baseline_result['scores'][0]) * 100,
                'rank_corr': compute_rank_correlation(baseline_result['indices'], var_result['indices'])
            }
            all_metrics.append(metrics)
            print(f"  {var_name}: Overlap@10={metrics['overlap@10']:.1f}%, ScoreRet={metrics['score_retention']:.1f}%")

        # 生成对比图
        output_path = op.join(output_base_dir, f"{case['name']}_comparison.png")
        plot_comparison_figure(
            baseline_result, variant_results, variant_names,
            output_path, topk=10
        )

    # 生成汇总表格
    df = pd.DataFrame(all_metrics)

    # 按变种类型汇总
    summary = df.groupby('variant').agg({
        'overlap@1': 'mean',
        'overlap@5': 'mean',
        'overlap@10': 'mean',
        'score_retention': 'mean',
        'rank_corr': 'mean'
    }).round(2)

    # 保存结果
    df.to_csv(op.join(output_base_dir, 'detailed_metrics.csv'), index=False)
    summary.to_csv(op.join(output_base_dir, 'summary_by_variant.csv'))

    print(f"\n{'=' * 60}")
    print("SUMMARY BY VARIANT TYPE")
    print(f"{'=' * 60}")
    print(summary.to_string())

    # 计算总体平均
    print(f"\n{'=' * 60}")
    print("OVERALL AVERAGE")
    print(f"{'=' * 60}")
    print(f"Overlap@1:  {df['overlap@1'].mean():.1f}%")
    print(f"Overlap@5:  {df['overlap@5'].mean():.1f}%")
    print(f"Overlap@10: {df['overlap@10'].mean():.1f}%")
    print(f"Score Retention: {df['score_retention'].mean():.1f}%")
    print(f"Rank Correlation: {df['rank_corr'].mean():.3f}")

    return df, summary


if __name__ == "__main__":
    config_file = 'logs/ICFG-PEDES/sdm+itc+aux_cnum3/configs.yaml'
    args = load_train_configs(config_file)
    args.batch_size = 512
    args.training = False
    device = "cuda"

    test_img_loader, test_txt_loader, num_class = build_dataloader(args)
    model = build_model(args, num_class)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())
    qfeats = F.normalize(qfeats, p=2, dim=1)
    gfeats = F.normalize(gfeats, p=2, dim=1)

    dataset = ICFGPEDES(root="/media/jqzhu/哈斯提·基拉/UniMoESE/data")
    test_dataset = dataset.test

    # 运行对比测试
    output_dir = "robustness_comparison_results"
    df, summary = run_robustness_comparison(
        model=model,
        gfeats=gfeats,
        gids=gids,
        img_paths=test_dataset['img_paths'],
        output_base_dir=output_dir,
        device=device
    )
