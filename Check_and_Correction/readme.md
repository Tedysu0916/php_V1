# Checking and Correction on 2026/01/13

## 1 Checking

We carefully checked our experimental code and performed various comparative validations to demonstrate the reliability of our results. The whole validation results are publicly released at https://github.com/Tedysu0916/php_V1/tree/main/Check_and_Correction. The main verification process and outcomes are summarized as follows.

**Table 1: Evaluation Using PHP and IRRA Codes on ICFG-PEDES [1].**

| Data Loading/Evaluation Code | .Pth file | Rank-1 (%) | mAP (%) |
|:----------------------------:|:---------:|:----------:|:-------:|
| PHP                          | PHP       | 97.32      | 78.16   |
| IRRA                         | PHP       | 97.32      | 78.16   |
| IRRA                         | IRRA      | 63.42      | 38.04   |
| PHP                          | IRRA      | 63.42      | 38.04   |

- **Evaluating Our PHP '.pth' in IRRA's Official Project.** We downloaded IRRA's [2] official code from https://github.com/anosorae/IRRA and replaced our data loading module (./datasets) and evaluation functions (./utils/metrics.py) with IRRA's implementations. The modified code is available at https://github.com/Tedysu0916/php_V1/tree/main/Check_and_Correction/Check/Our_PHP%E2%80%98.pth%E2%80%99_Evaluation_in_IRRA. Re-evaluating our PHP-trained .pth file yielded consistent results (Rank-1: 97.32%), as shown in Table 1. The ICFG-PEDES dataset statistics in the log also matched IRRA's official output.

- **Evaluating IRRA's '.pth' in Our PHP Project.** We downloaded IRRA's trained .pth file from https://drive.google.com/file/d/1Y3D7zZsKPpuEHWJ9nVecUW-HaKdjDI9g/view?usp=share_link, replaced our model folder with IRRA's implementation, and evaluated using IRRA's pre-trained weights. The code is available at https://github.com/Tedysu0916/php_V1/tree/main/Check_and_Correction/Check/IRRA%E2%80%98.pth%E2%80%99_Evaluation_in_Our_PHP_Project. The resulting Rank-1 accuracy of 63.42% aligns with IRRA's reported performance (Table 1), confirming the correctness of our data preparation, loading, and evaluation protocols.

- **Training Our PHP Architecture in IRRA's Official Project.** To eliminate potential issues from non-model factors, we integrated PHP's model folder into IRRA's official codebase (https://github.com/Tedysu0916/php_V1/tree/main/Check_and_Correction/Check/Our_PHP_Architecture_Training_in_IRRA_Official_Project), making only minimal modifications required by the architecture (e.g., optimizer and parameter configurations). After carefully verifying the absence of data leakage, we trained PHP within IRRA's framework and successfully reproduced results (Rank-1: 97.33%, mAP: 78.20%) that closely match those reported in our manuscript (Rank-1: 97.32%, mAP: 78.16%).

## 2 Correction

Our PHP supports two operating modes: parameter non-shared (NS) and shared (S), as shown in Figure 1. We discovered that our evaluation of the NS and S settings (TableIV in the manuscript submitted to *** Journal) was incorrect. This error was caused by an unexpected number of non-shared modules. To address this issue, we implemented a second version on 2026-01-13 and used it to generate the corrected results reported in Table 2. The codebase is available at (https://github.com/Tedysu0916/php_V1/tree/main/Check_and_Correction/Correction); it is a copy of our V2 (12/31/2025) with the correct configuration to enable easy reproduction.

As shown in Table 2, we observe substantial performance improvements when PHP operates with parameter sharing (S) between text and visual branches, while the non-sharing (NS) mode yields modest or even degraded performance. Taking ICFG-PEDES as an example, the parameter sharing mode achieves 97.32% R1 and 78.12% mAP, representing improvements of 75.56% and 63.96% over the non-sharing mode (21.76% and 14.16%). This finding aligns with our original motivation and intuition. In the non-sharing setting, cross-modal alignment between textual and visual features relies on two independent branches. Although this design provides each branch with greater learning flexibility, it also makes the model more susceptible to cross-modal noise—for example, images often contain large regions of irrelevant background, while descriptions may lack specificity. Such noise introduces undesirable signals and hinders accurate fine-grained cross-modal alignment, ultimately degrading the alignment of the enhanced features and reducing overall performance. In contrast, in the sharing setting, enforcing parameter sharing between the two branches appropriately constrains this flexibility, encouraging the model to capture semantically consistent patterns across modalities. This leads to more stable and effective alignment and improves overall performance. This observation is also consistent with the classic Occam's razor principle in machine learning: do not increase model complexity unless necessary.

**Figure 1: The illustration of the parameter-sharing setting switch (see the transparent blue region).**
![](images/NS vs S.png)

**Table 2: Performance (%) comparison of PHP's parameter non-shared (NS) and shared (S) modes.**

| Metric | ICFG-PEDES [1] |       | CUHK-PEDES [3] |       | RSTPReid [4] |       |
|:------:|:--------------:|:-----:|:--------------:|:-----:|:------------:|:-----:|
|        | NS             | S     | NS             | S     | NS           | S     |
| R1     | 21.76          | 97.32 | 50.29          | 82.80 | 38.20        | 76.70 |
| R5     | 39.12          | 99.38 | 74.58          | 93.55 | 63.85        | 92.85 |
| mAP    | 14.16          | 78.12 | 47.79          | 75.40 | 33.16        | 63.47 |

We sincerely thank all colleagues for their interest in our work and welcome any inquiries or feedback that further improve our method.

## References

[1] Z. Ding, C. Ding, Z. Shao, D. Tao, Semantically self-aligned network for text-to-image part-aware person re-identification, arXiv preprint arXiv:2107.12666 (2021).

[2] D. Jiang, M. Ye, Cross-modal implicit relation reasoning and aligning for text-to-image person retrieval, in: IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 2787–2797.

[3] S. Li, T. Xiao, H. Li, B. Zhou, D. Yue, X. Wang, Person search with natural language description, in: IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 1970–1979.

[4] A. Zhu, Z. Wang, Y. Li, X. Wan, J. Jin, T. Wang, F. Hu, G. Hua, Dssl: Deep surrounding-person separation learning for text-based person retrieval, in: ACM International Conference on Multimedia, 2021, pp. 209–217.
