# Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/anosorae/IRRA/blob/main/LICENSE) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-implicit-relation-reasoning-and/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=cross-modal-implicit-relation-reasoning-and)

Official PyTorch implementation of the paper Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval. (CVPR 2023) [arXiv](https://arxiv.org/abs/2303.12501)

### Requirements
we use single RTX3090 24G GPU for training and evaluation. 
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Guidance of experiments

You can download the checkpoints files which we reproduced from [Here](https://pan.baidu.com/s/1IXb0lQK8hPsIR0TYBRbRKQ?pwd=qwer) and then release them into ./logs, as follows. 
```
|-- logs/
|   |-- <ICFG-PEDES>/
|       |-- php_training_reproduce_on_IRRA
|           |--best.pth
|           |--configs.yaml
|           |--test_log.txt
|           |--train_log.txt
```

## Training
If you still concerned about our experiment, you could train the model yourself.
```python
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name php \
--img_aug \
--batch_size 64 \
--loss_names 'sdm+itc+aux' \
--dataset_name 'ICFG-PEDES' \
--root_dir "your data path" \
--num_epoch 60 \
--num_experts 4 \
--topk 2 \
--reduction 8 \
--moe_layers 4 \
--moe_heads 8 \
--transformer_lr_factor 1.0 \
--moe_lr_factor 2.0 \
--aux_factor 0.5 \
--lr 3e-6 \
--cnum 9 \
```


## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [TextReID](https://github.com/BrandonHanx/TextReID) and [TransReID](https://github.com/damo-cv/TransReID). We sincerely appreciate for their contributions.


## Citation
If you find this code useful for your research, please cite our paper.

```tex
@inproceedings{cvpr23crossmodal,
  title={Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval},
  author={Jiang, Ding and Ye, Mang},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```

## Contact
If you have any question, please feel free to contact us. E-mail: [jiangding@whu.edu.cn](mailto:jiangding@whu.edu.cn).
