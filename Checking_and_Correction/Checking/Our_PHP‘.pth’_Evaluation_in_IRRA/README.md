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
Download the CUHK-PEDES、ICFG-PEDES and RSTPReid dataset from [here](https://pan.baidu.com/s/1EtiFJBjjijhUD_mq5E1vMw?pwd=xvfy)

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

Please download the checkpoints files from [Here](https://pan.baidu.com/s/1siKkG7V74t_lTdlvrgmeow?pwd=qwer) and then release them into ./logs, as follows. 
```
|-- logs/
|   |-- <ICFG-PEDES>/
|       |-- sdm+itc+aux_cnum9
|           |--best.pth
|           |--configs.yaml
```

## Testing

```python
sh run_irra.sh
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
If you have any question, please feel free to contact us. E-mail: [jiangding@whu.edu.cn](mailto:jiangding@whu.edu.cn), [yemang@whu.edu.cn](mailto:yemang@whu.edu.cn).
