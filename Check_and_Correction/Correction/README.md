# Align What Truely Matters: Pedestrain-relevant Hierarchical Parsing Network for Text-based Person Retrieval
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/anosorae/IRRA/blob/main/LICENSE) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-implicit-relation-reasoning-and/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=cross-modal-implicit-relation-reasoning-and)

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

You can download the checkpoints files which we reproduced from [Here](https://pan.baidu.com/s/1e7DDZ1U5yTDAwXykEA_glg?pwd=qwer) and then release them into ./logs, as follows. 
```
|-- logs/
|   |-- <CUHK-PEDES>/
|       |-- 20260111_222604_php_ns
|           |--best.pth
|           |--configs.yaml
|           |--test_log.txt
|           |--train_log.txt
|       |-- sdm+itc+aux_cnum9
|
|   |-- <ICFG-PEDES>/
|       |-- 20260111_221755_php_ns
|           |--best.pth
|           |--configs.yaml
|           |--test_log.txt
|           |--train_log.txt
|       |-- sdm+itc+aux_cnum9
|
|   |-- <RSTPReid>/
|       |-- 20260111_222738_php_ns
|           |--best.pth
|           |--configs.yaml
|           |--test_log.txt
|           |--train_log.txt
|       |-- sdm+itc+aux_cnum9
```
## Testing
```
sh run.sh (for Share-Parameter PHP)
sh run_noshare.sh (for no Share-Parameter PHP)
```

Then you can get the results in the Table below.

![](images/figure1.png)

## Training

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

## Contact
If you have any question, please feel free to contact us. E-mail: [jiajunsu@hqu.edu.cn](mailto:jiajunsu@hqu.edu.cn), [jqzhu@hqu.edu.cn](mailto:jqzhu@hqu.edu.cn).
