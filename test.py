from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
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
    # args.root_dir = '/media/jqzhu/哈斯提·基拉/UniMoESE/data'
    # args.output_dir = '/media/jqzhu/哈斯提·基拉/UniMoESE/logs/CUHK-PEDES/sdm+itc+aux_cnum9'
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    model.eval()

    results = do_inference(model, test_img_loader, test_txt_loader)


    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    #
    # # 计算推理时间
    # end_time = time.time()
    # inference_time = end_time - start_time
    #
    # # 计算统计信息
    # num_images = len(test_img_loader.dataset)
    # num_texts = len(test_txt_loader.dataset)
    #
    # # 打印时间统计
    # logger.info("=" * 60)
    # logger.info("Inference Time Statistics:")
    # logger.info(f"Total inference time: {inference_time:.2f} seconds")
    # logger.info(f"Total inference time: {inference_time / 60:.2f} minutes")
    # logger.info(f"Number of images: {num_images}")
    # logger.info(f"Number of texts: {num_texts}")
    # logger.info(f"Average time per image: {inference_time / num_images:.4f} seconds")
    # logger.info(f"Images processed per second: {num_images / inference_time:.2f}")
    # logger.info(f"Text processed per second: {num_texts / inference_time:.2f}")
    # logger.info("=" * 60)
    #
    # # 也可以用PrettyTable展示
    # table = PrettyTable()
    # table.field_names = ["Metric", "Value"]
    # table.add_row(["Total Time (seconds)", f"{inference_time:.2f}"])
    # table.add_row(["Total Time (minutes)", f"{inference_time / 60:.2f}"])
    # table.add_row(["Number of Images", num_images])
    # table.add_row(["Number of Texts", num_texts])
    # table.add_row(["Avg Time per Image (sec)", f"{inference_time / num_images:.4f}"])
    # table.add_row(["Images per Second", f"{num_images / inference_time:.2f}"])
    #
    # logger.info("\nInference Time Summary:")
    # logger.info("\n" + str(table))
    # do_inference(model, test_img_loader, test_txt_loader)