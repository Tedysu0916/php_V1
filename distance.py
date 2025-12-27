import argparse
import os
import os.path as op

import pandas as pd
import torch
import numpy as np
import random
import time

from tqdm import tqdm

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs, load_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch.nn.functional as F
from scipy import stats

os.environ['CUDA_VISIBLE_DEVICES']='1'

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_distance_distributions(i_feat, t_feat, pids):
    """
    计算类内距离和类间距离的分布

    参数:
        i_feat: 图像特征 [bs, dim]
        t_feat: 文本特征 [bs, dim]
        pids: person IDs [bs]

    返回:
        intra_distances: 类内距离列表（同ID的图像-文本对）
        inter_distances: 类间距离列表（不同ID的图像-文本对）
    """
    # 归一化特征（使用余弦距离）
    i_feat = F.normalize(i_feat, p=2, dim=1)
    t_feat = F.normalize(t_feat, p=2, dim=1)

    intra_distances = []
    inter_distances = []

    # 计算所有图像-文本对的距离
    for i in range(len(pids)):
        for j in range(len(pids)):
            # 计算欧氏距离
            dist = torch.norm(i_feat[i] - t_feat[j]).item()

            if pids[i] == pids[j]:
                intra_distances.append(dist)
            else:
                inter_distances.append(dist)

    return intra_distances, inter_distances


def compute_similarity_distributions(i_feat, t_feat, pids):
    """
    计算类内相似度和类间相似度的分布（使用余弦相似度）

    参数:
        i_feat: 图像特征 [bs, dim]
        t_feat: 文本特征 [bs, dim]
        pids: person IDs [bs]

    返回:
        intra_similarities: 类内相似度列表
        inter_similarities: 类间相似度列表
    """
    # 归一化特征
    i_feat = F.normalize(i_feat, p=2, dim=1)
    t_feat = F.normalize(t_feat, p=2, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(i_feat, t_feat.t())  # [bs, bs]

    intra_similarities = []
    inter_similarities = []

    for i in range(len(pids)):
        for j in range(len(pids)):
            sim = similarity_matrix[i, j].item()

            if pids[i] == pids[j]:
                intra_similarities.append(sim)
            else:
                inter_similarities.append(sim)

    return intra_similarities, inter_similarities


def plot_distance_distribution(intra_distances, inter_distances, save_path, batch_idx, metric_type='distance'):
    """
    绘制类内和类间距离/相似度的分布图

    参数:
        intra_distances: 类内距离/相似度列表
        inter_distances: 类间距离/相似度列表
        save_path: 保存路径
        batch_idx: batch索引
        metric_type: 'distance' 或 'similarity'
    """
    plt.figure(figsize=(10, 6), dpi=150)

    # 设置字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5

    # 计算统计信息
    intra_mean = np.mean(intra_distances)
    inter_mean = np.mean(inter_distances)
    intra_std = np.std(intra_distances)
    inter_std = np.std(inter_distances)

    # 绘制直方图和核密度估计
    if metric_type == 'distance':
        # 距离：类内应该小，类间应该大
        intra_color = '#6BB6FF'  # 蓝色
        inter_color = '#FF6B6B'  # 红色
        xlabel = 'Feature Distance'
        title_prefix = 'Image-to-Text'
    else:
        # 相似度：类内应该大，类间应该小
        intra_color = '#6BB6FF'
        inter_color = '#FF6B6B'
        xlabel = 'Cosine Similarity'
        title_prefix = 'Image-to-Text'

    # 绘制类内分布
    plt.hist(intra_distances, bins=50, alpha=0.6, color=intra_color,
             label='Intra-class', density=True, edgecolor='black', linewidth=0.5)

    # 绘制类间分布
    plt.hist(inter_distances, bins=50, alpha=0.6, color=inter_color,
             label='Inter-class', density=True, edgecolor='black', linewidth=0.5)

    # 添加核密度估计曲线
    try:
        intra_kde = stats.gaussian_kde(intra_distances)
        inter_kde = stats.gaussian_kde(inter_distances)

        x_range = np.linspace(min(min(intra_distances), min(inter_distances)),
                              max(max(intra_distances), max(inter_distances)), 300)

        plt.plot(x_range, intra_kde(x_range), color=intra_color,
                 linewidth=2.5, alpha=0.8)
        plt.plot(x_range, inter_kde(x_range), color=inter_color,
                 linewidth=2.5, alpha=0.8)
    except:
        pass

    # 添加均值线
    plt.axvline(intra_mean, color=intra_color, linestyle='--',
                linewidth=2, alpha=0.8)
    plt.axvline(inter_mean, color=inter_color, linestyle='--',
                linewidth=2, alpha=0.8)

    # 添加文本标注
    y_max = plt.gca().get_ylim()[1]

    # 类内均值标注
    plt.text(intra_mean, y_max * 0.85, f'Intra Peak: {intra_mean:.2f}',
             color=intra_color, fontsize=13, fontweight='bold',
             ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                    facecolor='white', edgecolor=intra_color, linewidth=2))

    # 类间均值标注
    plt.text(inter_mean, y_max * 0.95, f'Inter Peak: {inter_mean:.2f}',
             color=inter_color, fontsize=13, fontweight='bold',
             ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                    facecolor='white', edgecolor=inter_color, linewidth=2))

    # 添加分离度标注（两个峰值之间）
    separation = abs(inter_mean - intra_mean)
    mid_point = (intra_mean + inter_mean) / 2
    plt.annotate('', xy=(inter_mean, y_max * 0.7), xytext=(intra_mean, y_max * 0.7),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(mid_point, y_max * 0.75, f'{separation:.2f}',
             fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                       edgecolor='black', linewidth=1.5, alpha=0.7))

    # 设置标签和标题
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title(f'{title_prefix} {metric_type.capitalize()} Distribution',
              fontsize=16, fontweight='bold', pad=15)

    # 添加图例
    plt.legend(loc='upper left', fontsize=13, frameon=True,
               facecolor='white', edgecolor='black', framealpha=0.9)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    filename = f'{metric_type}_distribution_batch_{batch_idx}.png'
    plt.savefig(os.path.join(save_path, filename),
                bbox_inches='tight', dpi=150)
    plt.close()

    # 打印统计信息
    print(f"\n{'=' * 60}")
    print(f"Batch {batch_idx} - {metric_type.upper()} Statistics:")
    print(f"Intra-class {metric_type}: {intra_mean:.4f} ± {intra_std:.4f}")
    print(f"Inter-class {metric_type}: {inter_mean:.4f} ± {inter_std:.4f}")
    print(f"Separation: {separation:.4f}")
    if metric_type == 'distance':
        print(f"Inter/Intra Ratio: {inter_mean / intra_mean:.4f}")
    else:
        print(f"Intra/Inter Ratio: {intra_mean / inter_mean:.4f}")
    print(f"{'=' * 60}\n")


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Feature Distance Analysis")
    parser.add_argument("--config_file",
                        default='logs/CUHK-PEDES/sdm+itc+aux_cnum9/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    args.training = True
    args.batch_size = 1024  # 可以根据需要调整
    train_loader, test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    model.eval()

    # 创建保存目录
    save_dir = './result'
    save_path_distance = './result/distance_distribution'
    save_path_similarity = './result/similarity_distribution'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_path_distance, exist_ok=True)
    os.makedirs(save_path_similarity, exist_ok=True)

    # 用于累积所有batch的统计
    all_intra_distances = []
    all_inter_distances = []
    all_intra_similarities = []
    all_inter_similarities = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_loader)):
            # 将batch移动到指定设备
            batch = {k: v.to(device) for k, v in batch.items()}
            img = batch['images'].to(device)
            caption = batch['caption_ids'].to(device)
            pid = batch['pids']

            # 提取特征
            i_feat = model.encode_image(img)
            t_feat = model.encode_text(caption)

            # 计算距离分布
            intra_dist, inter_dist = compute_distance_distributions(i_feat, t_feat, pid)

            # 计算相似度分布
            intra_sim, inter_sim = compute_similarity_distributions(i_feat, t_feat, pid)

            # 累积统计
            all_intra_distances.extend(intra_dist)
            all_inter_distances.extend(inter_dist)
            all_intra_similarities.extend(intra_sim)
            all_inter_similarities.extend(inter_sim)

            # 绘制当前batch的分布图
            plot_distance_distribution(intra_dist, inter_dist,
                                       save_path_distance, idx,
                                       metric_type='distance')

            plot_distance_distribution(intra_sim, inter_sim,
                                       save_path_similarity, idx,
                                       metric_type='similarity')

            # 只处理前几个batch（可选，避免生成太多图）
            if idx >= 4:  # 处理前5个batch
                break

    # 绘制所有batch累积的总体分布
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS (All Batches Combined)")
    print("=" * 60)

    plot_distance_distribution(all_intra_distances, all_inter_distances,
                               save_path_distance, 'overall',
                               metric_type='distance')

    plot_distance_distribution(all_intra_similarities, all_inter_similarities,
                               save_path_similarity, 'overall',
                               metric_type='similarity')

    print("\n✓ Analysis completed! Check the following directories:")
    print(f"  - Distance distributions: {save_path_distance}")
    print(f"  - Similarity distributions: {save_path_similarity}")
