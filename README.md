# Align What Truely Matters: Pedestrain-relevant Hierarchical Parsing Network for Text-based Person Retrieval
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/anosorae/IRRA/blob/main/LICENSE) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-implicit-relation-reasoning-and/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=cross-modal-implicit-relation-reasoning-and)

Official PyTorch implementation of the paper Align What Truely Matters: Pedestrain-relevant Hierarchical Parsing Network for Text-based Person Retrieval.

## Updates
- (2/10/2026) Code released!
- (5/8/2026)  Several experiment update!


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


## Beta Ablation Experiment

We conduct ablation experiments on the auxiliary loss weight by adjusting the `--aux_factor` parameter in [utils/options.py](utils/options.py#L81) (default: 0.5). This parameter controls the contribution of the load-balancing auxiliary loss $\mathcal{L}_{aux}$ during training. By varying `--aux_factor`, we can observe its impact on expert utilization balance and final retrieval performance. The experimental results are shown below:

![](images/beta.png)

## Shannon Entropy Expert Distribution Visualization

To analyze the expert balancing effect of the auxiliary loss, we visualize the expert selection distribution using Shannon entropy. The workflow is as follows:

1. During testing, [model/moe.py](model/moe.py#L49-L57) accumulates the number of times each expert is selected across all tokens via `expert_count_accum`.
2. After text/image feature extraction completes, [utils/metrics.py](utils/metrics.py#L48-L74) normalizes these counts into probability distributions and computes the Shannon entropy for each MoE layer — the higher the entropy, the better the expert balance.
3. The printed statistics are then filled into [Shannon_entropy_vis.py](Shannon_entropy_vis.py) to generate radar charts comparing expert distributions **with** vs. **without** $\mathcal{L}_{aux}$, saved as `images/entropy.png`.

![](images/entropy.png)

## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA), [DM-Adapter](https://github.com/Liu-Yating/DM-Adapter). We sincerely appreciate for their contributions.
