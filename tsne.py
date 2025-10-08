import argparse
import os
import os.path as op

import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES']="0"  #卡
import torch
import numpy as np
import random
import time

from sklearn import manifold
from tqdm import tqdm

# from diffusion_models.resample import UniformSampler, LossSecondMomentResampler
# from diffusion_models.respace import SpacedDiffusion, space_timesteps
# from diffusion_models import gaussian_diffusion as gd
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

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# 过滤掉 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

color = [(0.1, 0.1, 0.1, 1.0),  # r, g, b
         (0.5, 0.5, 0.5, 1.0),
         (1.0, 0.6, 0.1, 1.0),
         (1.0, 0.0, 0.0, 1.0),
         (1.0, 0.1, 0.7, 1.0),
         (0.9, 0.2, 0.4, 1.0),
         (0.8, 0.2, 1.0, 1.0),
         (0.8, 0.3, 0.2, 1.0),
         (0.7, 0.5, 0.3, 1.0),
         (0.7, 0.9, 0.4, 1.0),
         (0.7, 0.3, 0.8, 1.0),
         (0.0, 1.0, 0.0, 1.0),
         (0.4, 1.0, 0.6, 1.0),
         (0.3, 0.8, 0.5, 1.0),
         (0.1, 0.8, 1.0, 1.0),
         (0.5, 0.7, 0.9, 1.0),
         (0.4, 0.8, 0.3, 1.0),
         (0.5, 0.7, 0.4, 1.0),
         (0.2, 0.6, 0.8, 1.0),
         (0.1, 0.1, 1.0, 1.0),
         (0.3, 0.3, 0.9, 1.0),
         (0.6, 0.1, 0.4, 1.0),

         (0.2, 0.2, 0.5, 1.0),
         (0.3, 0.4, 0.6, 1.0),
         (0.4, 0.1, 0.9, 1.0),
         (0.5, 0.4, 0.2, 1.0),
         (0.6, 0.7, 0.3, 1.0),
         (0.7, 0.5, 0.6, 1.0),
         (0.8, 0.4, 0.1, 1.0),
         (0.9, 0.2, 0.3, 1.0),
         (1.0, 0.9, 0.3, 1.0),
         (0.0, 0.5, 0.5, 1.0),
         (0.2, 0.8, 0.2, 1.0),
         (0.4, 0.9, 0.7, 1.0),
         (0.6, 0.3, 0.2, 1.0),
         (0.8, 0.2, 0.6, 1.0),
         (0.9, 0.1, 0.9, 1.0),
         (0.3, 0.2, 0.4, 1.0),
         (0.5, 0.6, 0.7, 1.0),
         (0.6, 0.4, 0.9, 1.0),
         (0.7, 0.3, 0.1, 1.0),
         (0.8, 0.5, 0.2, 1.0),

         (0.2, 0.2, 0.5, 1.0),
         (0.3, 0.4, 0.6, 1.0),
         (0.4, 0.1, 0.9, 1.0),
         (0.5, 0.4, 0.2, 1.0),
         (0.6, 0.7, 0.3, 1.0),
         (0.7, 0.5, 0.6, 1.0),
         (0.8, 0.4, 0.1, 1.0),
         (0.9, 0.2, 0.3, 1.0),
         (1.0, 0.9, 0.3, 1.0),
         (0.0, 0.5, 0.5, 1.0),
         (0.2, 0.8, 0.2, 1.0),
         (0.4, 0.9, 0.7, 1.0),
         (0.6, 0.3, 0.2, 1.0),
         (0.8, 0.2, 0.6, 1.0),
         (0.9, 0.1, 0.9, 1.0),
         (0.3, 0.2, 0.4, 1.0),
         (0.5, 0.6, 0.7, 1.0),
         (0.6, 0.4, 0.9, 1.0),
         (0.7, 0.3, 0.1, 1.0),
         (0.8, 0.5, 0.2, 1.0), (0.2, 0.2, 0.5, 1.0),
         (0.3, 0.4, 0.6, 1.0),
         (0.4, 0.1, 0.9, 1.0),
         (0.5, 0.4, 0.2, 1.0),
         (0.6, 0.7, 0.3, 1.0),
         (0.7, 0.5, 0.6, 1.0),
         (0.8, 0.4, 0.1, 1.0),
         (0.9, 0.2, 0.3, 1.0),
         (1.0, 0.9, 0.3, 1.0),
         (0.0, 0.5, 0.5, 1.0),
         (0.2, 0.8, 0.2, 1.0),
         (0.4, 0.9, 0.7, 1.0),
         (0.6, 0.3, 0.2, 1.0),
         (0.8, 0.2, 0.6, 1.0),
         (0.9, 0.1, 0.9, 1.0),
         (0.3, 0.2, 0.4, 1.0),
         (0.5, 0.6, 0.7, 1.0),
         (0.6, 0.4, 0.9, 1.0),
         (0.7, 0.3, 0.1, 1.0),
         (0.8, 0.5, 0.2, 1.0),

         (0.2, 0.2, 0.5, 1.0),
         (0.3, 0.4, 0.6, 1.0),
         (0.4, 0.1, 0.9, 1.0),
         (0.5, 0.4, 0.2, 1.0),
         (0.6, 0.7, 0.3, 1.0),
         (0.7, 0.5, 0.6, 1.0),
         (0.8, 0.4, 0.1, 1.0),
         (0.9, 0.2, 0.3, 1.0),
         (1.0, 0.9, 0.3, 1.0),
         (0.0, 0.5, 0.5, 1.0),
         (0.2, 0.8, 0.2, 1.0),
         (0.4, 0.9, 0.7, 1.0),
         (0.6, 0.3, 0.2, 1.0),
         (0.8, 0.2, 0.6, 1.0),
         (0.9, 0.1, 0.9, 1.0),
         (0.3, 0.2, 0.4, 1.0),
         (0.5, 0.6, 0.7, 1.0),
         (0.6, 0.4, 0.9, 1.0),
         (0.7, 0.3, 0.1, 1.0),
         (0.8, 0.5, 0.2, 1.0),

         ]  # R G B


def plot_embedding(X, y, z, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 8), dpi=100)

    # 定义两种鲜艳的颜色
    colors = ['#FF6B6B', '#4ECDC4']  # 珊瑚红和青绿色

    # 用于创建图例的标记
    image_plotted = False
    text_plotted = False

    for i in range(X.shape[0]):
        if z[i] == 1:  # 图像特征（圆形）
            if not image_plotted:
                plt.scatter(X[i, 0], X[i, 1], s=120,
                            color=colors[0], edgecolor='black',
                            marker='o', alpha=0.8, label='Image Features')
                image_plotted = True
            else:
                plt.scatter(X[i, 0], X[i, 1], s=120,
                            color=colors[0], edgecolor='black',
                            marker='o', alpha=0.8)
        else:  # 文本特征（三角形）
            if not text_plotted:
                plt.scatter(X[i, 0], X[i, 1], s=120,
                            color=colors[1], edgecolor='black',
                            marker='^', alpha=0.8, label='Text Features')
                text_plotted = True
            else:
                plt.scatter(X[i, 0], X[i, 1], s=120,
                            color=colors[1], edgecolor='black',
                            marker='^', alpha=0.8)

    # 添加图例
    plt.legend(loc='lower right', fontsize=20, frameon=True,  # 必须设为 True 才能显示边框
               facecolor='white',  # 白色背景
               edgecolor='black',)  # 黑色边框
               # shadow=False)

    # 添加标题（如果提供）
    if title:
        plt.title(title, fontsize=14, fontweight='bold')

    # 去除坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    # 添加网格（可选）
    # plt.grid(True, alpha=0.3)
# def plot_embedding(X, y, z, title=None):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#     plt.figure(figsize=(10, 8), dpi=100)
#     cx, cy = [], []
#
#     r = []
#     # print(X.shape[0])
#     sjjcolor = ['#FF6B6B','#4ECDC4']
#     for i in range(X.shape[0]):
#         cx.append(X[i, 0])
#         cy.append(X[i, 1])
#         if z[i] == 1:
#             # print(i, y[i])
#             # print("==================",i,y[i])
#             # plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='o')
#             plt.scatter(X[i, 0], X[i, 1], s=120, color=sjjcolor[0], edgecolor='black', marker='o')
#         else:
#             # plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='^')
#             plt.scatter(X[i, 0], X[i, 1], s=120, color=sjjcolor[1], edgecolor='black', marker='o')
import os.path as osp
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# # 设置中文字体
# plt.rcParams['font.family'] = 'Microsoft YaHei'
# plt.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/sdm+itc+aux_cnum9/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    # args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    # test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    args.training = True
    args.batch_size = 1024
    train_loader, test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    model.eval()

    diagonal_similarities = []
    save_dir = './result'
    save_path = './result/tsne_php'
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_loader)):
            # 将batch移动到指定设备
            batch = {k: v.to(device) for k, v in batch.items()}
            img = batch['images'].to(device)
            caption = batch['caption_ids'].to(device)
            pid = batch['pids']
            bs = img.shape[0]
            img_pid = torch.zeros(bs,device='cuda:0')
            text_pid = torch.ones(bs, device='cuda:0')
            # print(img_pid)
            # print(text_pid)
            # print(img.shape,caption.shape)
            # print(pid)

            labels = torch.cat((img_pid, text_pid), dim=0)
            labels = labels.to(torch.int)
            z1 = torch.ones(pid.shape)
            z2 = torch.zeros(pid.shape)
            z = torch.cat((z1, z2), 0)
            # 提取特征
            b, c, h, w = img.shape
            # img = img.view(-1, c, h, w)
            i_feat = model.encode_image(img)
            t_feat = model.encode_text(caption)
            # print()
            # b,t,c = i_feat.shape
            # i_feat = i_feat.mean(1)
            # print(i_feat.shape,t_feat.shape)
            a = labels.unique()

            for i in range(len(a)):
                for j in range(len(labels)):
                    if labels[j] == a[i]:
                        labels[j] = i
                # print(labels)
            from sklearn.decomposition import PCA

            # data_time.update(time.time() - end)
            out = torch.cat((i_feat, t_feat), 0)
            # 将数据转为numpy格式
            out_np = out.detach().cpu().numpy()

            # 使用PCA进行预处理，减少到50维
            pca = PCA(n_components=50)
            out_pca = pca.fit_transform(out_np)
            #   tsne = manifold.TSNE(n_components=2, init='pca')
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=30, learning_rate=100,
                                 n_iter=3000)
            X_tsne = tsne.fit_transform(out_pca)
            plot_embedding(X_tsne, labels, z)
            plt.xticks([])
            plt.yticks([])
            # print(f"saving {save_path}")
            plt.savefig(osp.join(save_path, 'tsne_{}.jpg'.format(idx)))

