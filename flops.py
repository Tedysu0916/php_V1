from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils import logger
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/ICFG-PEDES/sdm+itc+aux_cnum9/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    args.test_batch_size = 64
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    # args.training = True
    # train_loader, test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    model.eval()






    # =============== 计算FLOPs差异 ===============
    logger.info("=" * 70)
    logger.info("Calculating FLOPs for encode_image and encode_text...")

    # 创建测试输入
    text_input = torch.randint(0, 49407, (1, 77)).to(device)
    image_input = torch.randn(1, 3, 384, 128).to(device)

    logger.info(f"Test input shapes:")
    logger.info(f"  Text input: {text_input.shape}")
    logger.info(f"  Image input: {image_input.shape}")

    # 方法1: 使用thop库计算FLOPs（推荐）
    try:
        from thop import profile, clever_format


        # 创建包装类来隔离encoder的计算
        class ImageEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.encode_image(x)


        class TextEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.encode_text(x)


        # 计算encode_image的FLOPs
        image_wrapper = ImageEncoderWrapper(model).to(device)
        image_wrapper.eval()
        with torch.no_grad():
            image_flops, image_params = profile(
                image_wrapper,
                inputs=(image_input,),
                verbose=False
            )

        # 计算encode_text的FLOPs
        text_wrapper = TextEncoderWrapper(model).to(device)
        text_wrapper.eval()
        with torch.no_grad():
            text_flops, text_params = profile(
                text_wrapper,
                inputs=(text_input,),
                verbose=False
            )

        # 格式化结果
        image_flops_str, image_params_str = clever_format([image_flops, image_params], "%.3f")
        text_flops_str, text_params_str = clever_format([text_flops, text_params], "%.3f")

        # 记录结果
        logger.info("=" * 70)
        logger.info("FLOPs Analysis Results (using thop):")
        logger.info("-" * 70)
        logger.info(f"{'encode_image:':<20} FLOPs = {image_flops_str:<15} Params = {image_params_str}")
        logger.info(f"{'encode_text:':<20} FLOPs = {text_flops_str:<15} Params = {text_params_str}")
        logger.info("-" * 70)
        logger.info(f"{'FLOPs Difference:':<20} {image_flops - text_flops:,.0f}")
        logger.info(f"{'FLOPs Ratio:':<20} {image_flops / text_flops:.2f}x (image/text)")
        logger.info("=" * 70)

        # 详细分析
        logger.info("\nDetailed Analysis:")
        logger.info(f"  - Image encoder requires {(image_flops - text_flops) / 1e9:.2f} GFLOPs more than text encoder")
        logger.info(f"  - Image encoder computational cost is {image_flops / text_flops:.2f}x of text encoder")

        if image_flops > text_flops:
            logger.info("  - Image encoder has higher computational cost due to:")
            logger.info("    1. Larger spatial dimensions (384x128) creating more tokens")
            logger.info("    2. Vision transformer processing more patch embeddings")
            logger.info("    3. SAFL module operations on image features")

        # 保存FLOPs结果
        flops_results = {
            'image_flops': float(image_flops),
            'text_flops': float(text_flops),
            'difference': float(image_flops - text_flops),
            'ratio': float(image_flops / text_flops),
            'image_flops_formatted': image_flops_str,
            'text_flops_formatted': text_flops_str
        }

        import json

        flops_file = op.join(args.output_dir, 'flops_analysis.json')
        with open(flops_file, 'w') as f:
            json.dump(flops_results, f, indent=4)
        logger.info(f"\nFLOPs analysis results saved to: {flops_file}")

    except ImportError:
        logger.warning("thop library not installed. Please install it using: pip install thop")
        logger.warning("Trying alternative method with fvcore...")

        # 方法2: 使用fvcore作为备选
        try:
            from fvcore.nn import FlopCountAnalysis

            with torch.no_grad():
                # 计算encode_image的FLOPs
                image_flops_analyzer = FlopCountAnalysis(model.encode_image, image_input)
                image_flops = image_flops_analyzer.total()

                # 计算encode_text的FLOPs
                text_flops_analyzer = FlopCountAnalysis(model.encode_text, text_input)
                text_flops = text_flops_analyzer.total()

            logger.info("=" * 70)
            logger.info("FLOPs Analysis Results (using fvcore):")
            logger.info("-" * 70)
            logger.info(f"encode_image FLOPs: {image_flops:,.0f}")
            logger.info(f"encode_text FLOPs: {text_flops:,.0f}")
            logger.info(f"FLOPs Difference: {image_flops - text_flops:,.0f}")
            logger.info(f"FLOPs Ratio: {image_flops / text_flops:.2f}x")
            logger.info("=" * 70)

        except ImportError:
            logger.error("Neither thop nor fvcore is installed.")
            logger.error("Please install one of them:")
            logger.error("  pip install thop")
            logger.error("  pip install fvcore")

    # =============== 测试实际推理速度 ===============
    logger.info("\n" + "=" * 70)
    logger.info("Testing actual inference speed...")

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model.encode_image(image_input)
            _ = model.encode_text(text_input)

    # 测试encode_image速度
    num_iterations = 100
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model.encode_image(image_input)
    torch.cuda.synchronize()
    image_time = (time.time() - start_time) / num_iterations

    # 测试encode_text速度
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model.encode_text(text_input)
    torch.cuda.synchronize()
    text_time = (time.time() - start_time) / num_iterations

    logger.info(f"Average inference time (over {num_iterations} iterations):")
    logger.info(f"  encode_image: {image_time * 1000:.2f} ms/batch")
    logger.info(f"  encode_text: {text_time * 1000:.2f} ms/batch")
    logger.info(f"  Speed ratio: {image_time / text_time:.2f}x (image/text)")
    logger.info(f"  Image FPS: {1 / image_time:.2f}")
    logger.info(f"  Text FPS: {1 / text_time:.2f}")
    logger.info("=" * 70)

# # =============== 可选：分析各个模块的FLOPs贡献 ===============
# if hasattr(model, 'base_model') and hasattr(model, 'safl'):
#     logger.info("\n" + "=" * 70)
#     logger.info("Component-wise FLOPs analysis...")
#
#     try:
#         from thop import profile
#
#         # 测试base_model的image encoder
#         with torch.no_grad():
#             base_image_features = model.base_model.encode_image(image_input)
#             if len(base_image_features.shape) == 3:
#                 # 去除CLS token，只保留patch tokens
#                 patch_features = base_image_features[:, 1:, :]
#
#                 # 测试SAFL模块
#                 safl_flops, _ = profile(model.safl, inputs=(patch_features,), verbose=False)
#                 safl_flops_str, _ = clever_format([safl_flops, 0], "%.3f")
#                 logger.info(f"SAFL module FLOPs: {safl_flops_str}")
#
#             # 测试shared_transformer
#             if hasattr(model, 'shared_transformer'):
#                 dummy_features = torch.randn_like(patch_features)
#                 trans_flops, _ = profile(
#                     lambda x: model.shared_transformer(x, 0)[0],
#                     inputs=(dummy_features,),
#                     verbose=False
#                 )
#                 trans_flops_str, _ = clever_format([trans_flops, 0], "%.3f")
#                 logger.info(f"Shared Transformer FLOPs: {trans_flops_str}")
#
#     except Exception as e:
#         logger.warning(f"Component-wise analysis failed: {str(e)}")
#
# logger.info("=" * 70)
# logger.info("FLOPs analysis completed!")

# # =============== 继续原有的测试流程 ===============
# logger.info("\nContinuing with original inference...")

# 这里可以添加原本的推理代码，比如：
# test(args, model, test_img_loader, test_txt_loader)
