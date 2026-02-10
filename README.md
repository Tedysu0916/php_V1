# Align What Truely Matters: Pedestrain-relevant Hierarchical Parsing Network for Text-based Person Retrieval
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/anosorae/IRRA/blob/main/LICENSE) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-implicit-relation-reasoning-and/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=cross-modal-implicit-relation-reasoning-and)

Official PyTorch implementation of the paper Align What Truely Matters: Pedestrain-relevant Hierarchical Parsing Network for Text-based Person Retrieval.

## Updates
- (2/10/2026) Code released!



## Highlights
In this paper, we propose a Pedestrian-relevant Hierarchical Parsing (PHP) module to extract well-aligned fine-grained visual and textual features for alignment. First, we design a Coarse Relevant Feature Mapping (CRFM) module, which uses learnable unified tokens to project both modalities into a shared low-dimensional space, enabling coarse-level semantic filtering. Then, we design an Expert-driven Feature Parsing (EFP) module that integrates the representational power of mixture of experts with a modality-aware gating mechanism to uncover deep semantic associations between text and image features. Both the CRFM and EFP modules share parameters across the two branches, which facilitates the acquisition of cross-modal semantically aligned information.
![](images/overview_new.png)

## Usage
### Requirements

we use single RTX3090 24G GPU for training and evaluation. 
```
pytorch 2.1.1
torchvision 0.16.1
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


## Testing

```
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

## Guidance of experiments

Please download the checkpoints files from [here](https://pan.baidu.com/s/1ortefna6rc186hzTL5-OTw?pwd=qwer) and then release them into ./logs, as follows. 
```
|-- logs/
|   |-- <CUHK-PEDES>/
|       |-- itc+aux
|           |--best.pth
|           |--configs.yaml
|       |-- sdm+aux
|       |-- ...
|       |-- sdm+itc+aux_cnum3
|       |-- ...
|
|   |-- <ICFG-PEDES>/
|       |-- itc+aux
|           |--best.pth
|           |--configs.yaml
|       |-- sdm+aux
|       |-- ...
|       |-- sdm+itc+aux_cnum3
|       |-- ...
|
|   |-- <RSTPReid>/
|       |-- itc+aux
|           |--best.pth
|           |--configs.yaml
|       |-- sdm+aux
|       |-- ...
|       |-- sdm+itc+aux_cnum3
|       |-- ...
```
### Reproducing Results
To verify and reproduce our experimental results, simply modify the config file path in the testing command:
```
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--config_file 'xxxx.yaml'
```

### SOTA Table
![](images/c1.png)
![](images/c0.png)

**The above report php's result among three datasets are corresponding to "./sdm+itc+aux_cnum9".**
### File Naming Convention
- **Loss Ablation Experiments**: Each subdirectory represents different loss function combinations:
![](images/s1.png)
  - Files **without** `_cnum` suffix use the default setting of `cnum=9`
  - `itc+aux`: Image-Text Contrastive loss with auxiliary loss
  - `sdm+aux`: Similarity Distribution Matching loss with auxiliary loss
  - `sdm+itc+aux`: Combined SDM and ITC losses with auxiliary loss

- **Model ablation**:
![](images/cp3.png)
  - Files with **"ablation_base"** denotes only use baseline(without CRFM and EFP). Besides, you should change the encoder to model.encode_text_base/model.encode_image_base in ./utils/metrics.py.
  - Files with **"ablation_crfm"** denotes only use CRFM model. Besides, you should change the encoder to model.encode_text_crfm/model.encode_image_crfm in ./utils/metrics.py.
  - Files with **"ablation_efp"** denotes only use CRFM model. Besides, you should change the encoder to model.encode_text_efp/model.encode_image_efp in ./utils/metrics.py.
  - Files with **"sdm+itc+aux_cnum9"** denotes the whole php model. Besides, you should change the encoder to model.encode_text/model.encode_image in ./utils/metrics.py

## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA), [DM-Adapter](https://github.com/Liu-Yating/DM-Adapter). We sincerely appreciate for their contributions.

## Contact
If you have any question, please feel free to contact us. E-mail: [jiajunsu@hqu.edu.cn](mailto:jiajunsu@hqu.edu.cn).
