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

from utils.simple_tokenizer import SimpleTokenizer


def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    image_paths = [img_paths[j] for j in indices[idx]]
    image_ids = gids[indices[idx]]
    gt_image_idx = gt_img_paths.index(query_id)
    gt_image_path = img_paths[gt_image_idx]
    return query_id, image_ids, query_caption, image_paths, gt_image_path


def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path, output_dir,
                          method_name='default', show_gt=True):
    """保存检索结果图像到指定目录"""
    col = len(image_paths)

    if show_gt and gt_img_path is not None:
        fig = plt.figure(figsize=(20, 4))
        plt.suptitle(f"Query: {query_caption}", fontsize=12, wrap=True)

        # 绘制真实图像
        plt.subplot(1, col + 1, 1)
        img = Image.open(gt_img_path)
        img = img.resize((128, 256))
        plt.imshow(img)
        plt.title("Ground Truth", fontsize=10)
        plt.xticks([])
        plt.yticks([])
        start_idx = 2
    else:
        fig = plt.figure(figsize=(18, 4))
        plt.suptitle(f"Query: {query_caption}", fontsize=12, wrap=True)
        start_idx = 1

    # 绘制检索结果
    for i in range(col):
        plt.subplot(1, col + (1 if show_gt else 0), start_idx + i)
        img = Image.open(image_paths[i])

        bwith = 2  # 边框宽度
        ax = plt.gca()
        if show_gt and gt_img_path and image_ids[i] == query_id:
            ax.spines['top'].set_color('lawngreen')
            ax.spines['right'].set_color('lawngreen')
            ax.spines['bottom'].set_color('lawngreen')
            ax.spines['left'].set_color('lawngreen')
            plt.title(f"Rank {i + 1} ✓", fontsize=10)
        else:
            ax.spines['top'].set_color('blue')
            ax.spines['right'].set_color('blue')
            ax.spines['bottom'].set_color('blue')
            ax.spines['left'].set_color('blue')
            plt.title(f"Rank {i + 1}", fontsize=10)

        img = img.resize((128, 256))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像
    result_path = op.join(output_dir, f"{method_name}_results.png")
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def encode_custom_text(model, text, tokenizer, device):
    """编码自定义文本"""
    # 根据你的模型实现调整tokenizer的使用方式
    # 这里假设模型有一个encode_text方法
    model.eval()
    with torch.no_grad():
        # 如果模型使用CLIP tokenizer
        # print(text)
        tokens = tokenize(text, tokenizer=SimpleTokenizer(), text_length=77, truncate=True).unsqueeze(0).to(device)
        text_features = model.encode_text(tokens)
        print(text_features.shape)
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features


def visualize_custom_query(custom_text, model, gfeats, gids, img_paths,
                           output_dir='noisy_query_results', method_name='custom',
                           topk=10, device='cuda'):
    """可视化自定义文本查询的结果"""

    # 创建输出目录
    if not op.exists(output_dir):
        os.makedirs(output_dir)


    text_feat = encode_custom_text(model, custom_text, None, device)

    # 计算相似度
    similarity = text_feat @ gfeats.t()  # 1 x N
    scores, indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=True)

    # 获取top-k图像
    indices = indices.cpu().numpy()[0]
    scores = scores.cpu().numpy()[0]
    image_paths = [img_paths[i] for i in indices]
    image_ids = gids[indices].cpu().numpy()

    # 保存查询文本
    with open(op.join(output_dir, "query_text.txt"), "w", encoding="utf-8") as f:
        f.write(custom_text)
        f.write("\n\nTop-10 Similarity Scores:\n")
        for i, score in enumerate(scores):
            f.write(f"Rank {i + 1}: {score:.4f}\n")

    # 可视化（不显示Ground Truth）
    plot_retrieval_images(
        query_id=None,
        image_ids=image_ids,
        query_caption=custom_text,
        image_paths=image_paths,
        gt_img_path=None,
        output_dir=output_dir,
        method_name=method_name,
        show_gt=False
    )

    print(f"\n自定义查询完成！")
    print(f"查询文本: {custom_text}")
    print(f"结果保存在: {output_dir}")
    print(f"\nTop-5 相似度分数:")
    for i, score in enumerate(scores[:5]):
        print(f"  Rank {i + 1}: {score:.4f}")


def visualize_all_test_results(indices, qids, gids, captions, img_paths, gt_img_paths,
                               output_base_dir='visualization_results', method_name='default'):
    """对整个测试集进行可视化处理"""
    # 创建输出基础目录
    if not op.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 使用tqdm显示进度
    for idx in tqdm.tqdm(range(len(qids)), desc="处理查询"):
        try:
            # 获取当前查询的信息
            query_id, image_ids, query_caption, image_paths, gt_img_path = get_one_query_caption_and_result_by_id(
                idx, indices, qids, gids, captions, img_paths, gt_img_paths
            )

            # 创建以ID命名的文件夹
            query_folder = op.join(output_base_dir, f"id_{query_id}")
            if not op.exists(query_folder):
                os.makedirs(query_folder)

            # 保存查询文本到文本文件
            with open(op.join(query_folder, "query_text.txt"), "w", encoding="utf-8") as f:
                f.write(query_caption)

            # 可视化检索结果并保存
            plot_retrieval_images(
                query_id, image_ids, query_caption, image_paths, gt_img_path,
                output_dir=query_folder, method_name=method_name
            )

        except Exception as e:
            print(f"处理查询 {idx} (ID: {qids[idx]}) 时出错: {e}")


# 主执行代码
if __name__ == "__main__":
    config_file = 'logs/ICFG-PEDES/sdm+itc+aux_cnum9/configs.yaml'
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

    # 计算相似度和排序结果
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())
    qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
    gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

    dataset = ICFGPEDES(root="/media/jqzhu/哈斯提·基拉/UniMoESE/data")
    test_dataset = dataset.test
    # ==================== 选择模式 ====================
    MODE = "custom"  # 改为 "all" 可视化整个测试集，改为 "custom" 进行自定义查询

    if MODE == "custom":
        # ==================== 自定义查询 ====================
        # custom_queries = [
        #     "black shirt man",
        #     "woman in red dress",
        #     "person wearing blue jeans"
        # ]
        custom_queries = [

            # 服装颜色和类型
            "white hoodie person",
            "woman in yellow jacket",
            "man wearing green t-shirt",
            "person in purple sweater",
            "gray coat woman",
            "striped shirt man",

            # 性别和年龄描述
            "young girl in pink",
            "elderly man walking",
            "teenage boy running",
            "middle-aged woman standing",

            # 配饰和细节
            "person with backpack",
            "woman carrying handbag",
            "man wearing hat",
            "person with glasses",
            "woman in high heels",
            "man with beard",

            # 组合描述
            "tall man in dark clothes",
            "short woman with long hair",
            "person in casual outfit",
            "formally dressed man"
        ]

        noisy_queries = [
            # 混合无关词汇
            "black shirt pizza man",
            "woman red dress computer mouse",
            "blue jeans person mountain bicycle",
            "walking elephant green jacket",
            "car window yellow shirt person",

            # 语法错误和拼写错误
            "persn wearing whit shirt",
            "womam in blur dress",
            "man shirt blak color",

            # 无意义组合
            "keyboard woman red dancing",
            "telephone man blue running shoes",
            "camera person wearing clouds",
            "microwave girl pink standing",

            # 完全无关的查询
            "cooking recipe ingredients",
            "weather forecast sunny",
            "programming language python",
            "mathematics equation solving",

            # 部分相关但有噪声
            "beautiful landscape woman red dress",
            "database query man black shirt",
            "internet connection person walking",
            "artificial intelligence woman glasses",

            # 重复和冗余词汇
            "man man wearing wearing black black shirt shirt",
            "woman woman woman in red red dress",
            "person person blue blue jeans jeans walking walking"
        ]

        for custom_text in noisy_queries:
            # 为每个查询创建单独的文件夹
            safe_name = custom_text.replace(" ", "_")
            output_dir = f"noisy_query_results/{safe_name}"

            # print('custom_text',custom_text)
            visualize_custom_query(
                custom_text=custom_text,
                model=model,
                gfeats=gfeats,
                gids=gids,
                img_paths=test_dataset['img_paths'],
                output_dir=output_dir,
                method_name="custom_query",
                topk=10,
                device=device
            )

    elif MODE == "all":
        # ==================== 可视化整个测试集 ====================
        similarity = qfeats @ gfeats.t()
        _, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)

        # 加载数据集
        dataset = ICFGPEDES(root="/media/jqzhu/哈斯提·基拉/UniMoESE/data")
        test_dataset = dataset.test

        img_paths = test_dataset['img_paths']
        captions = test_dataset['captions']
        gt_img_paths = test_dataset['image_pids']

        # 将indices移到与gids相同的设备上
        indices = indices.to(gids.device)

        output_dir = "visualization_results_irra"
        method_name = "php"

        visualize_all_test_results(
            indices, qids, gids, captions, img_paths, gt_img_paths,
            output_base_dir=output_dir, method_name=method_name
        )

        print(f"可视化完成！结果保存在 {output_dir} 目录中")