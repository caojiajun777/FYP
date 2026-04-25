<div align="center">

# EDA Schematic Repository Classification
### Dissertation Artefacts — Final Year Project

**EDA Provenance Identification and Schematic-Level Function Tagging for Mixed-Source PCB Schematic Repositories**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

**Author:** Cao Jiajun (曹佳竣)  
**Programme:** BEng Electrical & Electronic Engineering  
**Institution:** University of Nottingham Ningbo China (UNNC)  
**Academic Year:** 2025–2026

</div>

---

## Overview

This repository contains all code, result artefacts, and evaluation evidence for the dissertation.
The project addresses two complementary problems in mixed-source PCB schematic repositories:

| Task | Problem | Method | Key Result |
|---|---|---|---|
| **Task 1** | EDA Tool Provenance (5-class classification) | ViT-B/16 fine-tuning | **Acc 99.05%**, Macro-F1 99.15% |
| **Task 2** | Schematic Function Tagging (5-label multi-label) | Qwen2.5-VL-7B LoRA | **EM 0.5672**, Micro-F1 0.8566 |

---

## Repository Structure

```
FYP_repo/
├── task1_scripts/                          # Task 1 — EDA Provenance Identification
│   ├── train_vit.py                        # ★ Protocol A: ViT-B/16 training (final model)
│   ├── train_kfold.py                      # Protocol B: 3-fold CV (resnet50/vit_b_16/convnext)
│   ├── train_baselines.py                  # ResNet50 & ConvNeXt baseline training
│   ├── evaluate_vit_model.py               # Protocol A heldout test evaluation
│   ├── evaluate_vit_vs_resnet.py           # Cross-model comparison
│   ├── ablation_region.py                  # Input region ablation (ResNet50)
│   ├── ablation_region_complete.py         # Region ablation (ResNet50 + ConvNeXt)
│   ├── ablation_region_vit_comparison.py   # Region ablation (ViT vs ResNet comparison)
│   ├── ablation_input_representation.py    # RGB vs Grayscale / footer-mask ablation
│   ├── ablation_efficiency_benchmark.py    # Inference latency benchmark
│   ├── grad_cam.py                         # Grad-CAM visualisation (ResNet50 / ViT)
│   ├── occlusion_sensitivity.py            # Occlusion sensitivity analysis
│   ├── feature_visualization_vit.py        # t-SNE feature space visualisation
│   ├── analyze_errors.py                   # 3-fold CV error analysis
│   ├── analyze_task1_errors_gradcam.py     # Grad-CAM on misclassified samples
│   ├── compare_resnet_vit_errors.py        # ResNet vs ViT error comparison
│   ├── build_train_splits.py               # Fixed-split manifest builder
│   ├── clean_dataset.py                    # Dataset deduplication & cleaning
│   ├── verify_data_quality.py              # Perceptual-hash leakage check
│   ├── count_dups.py                       # Duplicate count utility
│   └── plot_training_curves.py             # Regenerate training curves from history.json
│
├── task1_results/                          # Task 1 result artefacts
│   ├── metrics/
│   │   ├── test_metrics.json               # ★ Acc=0.9905, Macro-F1=0.9915, n=317
│   │   ├── val_metrics.json                # Val metrics, n=316
│   │   ├── test_evaluation_report.md
│   │   ├── val_evaluation_report.md
│   │   └── README.md                       # Clarifies post-cleaning Protocol A version
│   ├── ablation/
│   │   ├── resnet_vs_vit_comparison.json   # ★ ViT Full=98.45%, ResNet Full=72.39%
│   │   ├── region_ablation_results.json
│   │   ├── complete_region_ablation_results.json
│   │   ├── ablation_results.json
│   │   ├── efficiency_benchmark_results.json
│   │   ├── footer_masked_supplement.json
│   │   └── ABLATION_COMPREHENSIVE_REPORT.md
│   ├── visualizations/
│   │   ├── confusion_matrix_test.png
│   │   ├── confusion_matrix_val.png
│   │   ├── train_val_curves.png
│   │   └── training_loss_only.png
│   ├── interpretability/
│   │   └── comprehensive_interpretability_report.md
│   ├── kfold_cv_results.md                 # ★ ResNet50=98.76%, ViT=97.80% (3-fold CV)
│   ├── data_cleaning_impact_report.md      # ★ 4320→3546, split 2913/316/317
│   ├── data_cleaning_report.md
│   ├── cleaning_statistics.json            # 2nd-round cleaning stats (3247→3230)
│   ├── CLEANING_STATISTICS_NOTE.md         # Disambiguates cleaning_statistics.json
│   ├── FINAL_MODEL_SELECTION_ANALYSIS.md
│   └── TASK1_HELDOUT_METRICS_SUMMARY.md
│
├── gold_standard/                          # Task 2 gold benchmark splits
│   ├── test_split.json                     # 134-image gold test set (human-verified)
│   ├── val_split.json                      # 100-image gold validation set
│   └── gold_val_test.json                  # Merged manifest (leakage guard)
│
├── lora_exports/
│   └── qwen2_5_vl_7b/
│       ├── checkpoint-675/                         # Best LoRA checkpoint (Git LFS)
│       ├── checkpoint-400/                         # Comparison checkpoint
│       ├── gold_test_predictions_ckpt675.json      # ★ EM=0.5672, Micro-F1=0.8566
│       └── gold_test_predictions_ckpt400.json      # Checkpoint-400 comparison
│
├── task2_vit_baseline/
│   ├── train_split.json                    # 2199-sample deduped training pool
│   ├── vit_test_metrics.json               # EM=0.2388, Micro-F1=0.6561
│   └── vit_test_predictions.json
│
├── task2_scripts/
│   ├── build_task2_vit_train_split.py      # Builds 2199-sample train split
│   ├── train_task2_vit_baseline.py         # ViT-B/16 multi-label training
│   ├── evaluate_gold_test.py               # Evaluates Qwen LoRA on gold test set
│   └── eval_decode_metrics.py              # Checkpoint comparison evaluation
│
├── qwen_train_high.json                    # 2271-entry silver-label pool
├── train_resnet_baseline.py                # ResNet50 5-label multi-label training
├── train_task2_qwen_vl_lora.py             # Qwen2.5-VL-7B LoRA fine-tuning
├── prepare_lora_dataset.py                 # Builds LLaMA-Factory training JSON
├── resnet50_gold_test_metrics.json         # ResNet50 gold test: EM=0.3358, Micro-F1=0.6921
└── archive/                                # Legacy documents (pre-final system)
```

> **Note on paths:** All scripts use `os.environ.get()` for data/model paths with sensible defaults.  
> Set `DATA_ROOT`, `IMAGE_ROOT`, `TASK1_DATA_ROOT`, `TASK1_KFOLD_DATA_ROOT`, `TASK1_VIT_OUTPUT_DIR`, `TASK1_KFOLD_MODEL_ROOT` as needed before running locally (see per-task sections below).

---

## Task 1 — EDA Provenance Identification

**5 classes:** Altium · KiCad · OrCAD · Eagle · JLC/EasyEDA  
**Dataset:** 4320 raw → 3546 after deduplication and cleaning  
**Protocol A fixed split:** train 2913 / val 316 / test 317  
**Protocol B:** stratified 3-fold cross-validation on the same 3546 images

**Final result (ViT-B/16, Protocol A, 317-image held-out test):**

| Metric | Value |
|---|---|
| Accuracy | **99.05%** |
| Macro-F1 | **99.15%** |

**Protocol B cross-validation (3-fold):**

| Model | Mean Accuracy |
|---|---|
| ResNet50 | 98.76% |
| ViT-B/16 | 97.80% |

**Ablation highlights (`task1_results/ablation/resnet_vs_vit_comparison.json`):**

| Configuration | ViT-B/16 | ResNet50 |
|---|---|---|
| Full image | 98.45% | 72.39% |
| Bottom region only | — | lower |
| Center region only | — | lower |

### Reproducing Task 1

```bash
# Set paths
export TASK1_DATA_ROOT=/path/to/EDA_cls_dataset          # train/val_cropped/test subdirs
export TASK1_KFOLD_DATA_ROOT=/path/to/EDA_cls_dataset_kfold
export TASK1_VIT_OUTPUT_DIR=/path/to/output/runs_vit/train_vit_b16_best
export TASK1_KFOLD_MODEL_ROOT=/path/to/output/runs_kfold

# 1. (Optional) Data cleaning
python task1_scripts/clean_dataset.py
python task1_scripts/verify_data_quality.py
python task1_scripts/build_train_splits.py

# 2. Protocol A: Train ViT-B/16 (final model)
python task1_scripts/train_vit.py

# 3. Protocol A: Evaluate on held-out test set
python task1_scripts/evaluate_vit_model.py

# 4. Protocol B: 3-fold cross-validation
python task1_scripts/train_kfold.py --model vit_b_16 --folds 3
python task1_scripts/train_kfold.py --model resnet50  --folds 3

# 5. Ablation studies
python task1_scripts/ablation_region_vit_comparison.py
python task1_scripts/ablation_input_representation.py
python task1_scripts/ablation_efficiency_benchmark.py

# 6. Interpretability
python task1_scripts/grad_cam.py
python task1_scripts/feature_visualization_vit.py
```

---

## Task 2 — Schematic-Level Function Tagging

**5 labels:** power · interface · communication · signal · control  
**Silver-label training pool:** 2271 candidates → 2199 unique (72 internal filename duplicates removed; 0 gold overlap)  
**Gold benchmark:** 100-image val split + 134-image test split (human-verified multi-label)

> `generate_silver_labels_qwen.py` used a 6-label prompt (including `eval_board`) during initial collection; `prepare_lora_dataset.py` strictly filters to the final 5-label vocabulary before training.

**Final benchmark (Qwen2.5-VL-7B LoRA, checkpoint-675, gold test n=134):**

| Metric | Value |
|---|---|
| Exact Match | **0.5672** |
| Micro-F1 | **0.8566** |
| Macro-F1 | **0.8561** |
| Format Errors | 0 |

**Baseline comparison:**

| Model | Exact Match | Micro-F1 | Macro-F1 |
|---|---|---|---|
| ResNet50 pure-vision | 0.3358 | 0.6921 | 0.6581 |
| ViT-B/16 pure-vision | 0.2388 | 0.6561 | 0.6229 |
| **Qwen2.5-VL-7B LoRA** | **0.5672** | **0.8566** | **0.8561** |

> ResNet50 metrics are transcribed from experimental logs (`TASK2_WORKLOG.md`); no raw prediction file exists for the ResNet50 baseline.

---

## Reproducing the Task 2 Experiments

### 1. Set environment variables

```bash
# Linux / Mac
export DATA_ROOT=/path/to/FYP_repo
export IMAGE_ROOT=/path/to/directory_containing_task2_images

# Windows PowerShell
$env:DATA_ROOT = "D:\path\to\FYP_repo"
$env:IMAGE_ROOT = "D:\path\to\data"
```

### 2. Build the ViT training split

```bash
python task2_scripts/build_task2_vit_train_split.py
```

### 3. Train the ViT-B/16 baseline

```bash
python task2_scripts/train_task2_vit_baseline.py
```

### 4. Fine-tune Qwen2.5-VL-7B LoRA (requires an A100/H100 GPU with ≥40 GB VRAM)

```bash
# Prepare LLaMA-Factory training data
python prepare_lora_dataset.py

# Run LoRA training (uses LLaMA-Factory)
python train_task2_qwen_vl_lora.py
```

### 5. Evaluate Qwen LoRA on the gold test set

```bash
python task2_scripts/evaluate_gold_test.py
```

---

## Key File Index for Audit / Reproducibility

### Task 1

| File | Role |
|---|---|
| `task1_results/metrics/test_metrics.json` | ★ Final test metrics (Acc=0.9905, Macro-F1=0.9915, n=317) |
| `task1_results/metrics/val_metrics.json` | Val metrics (n=316) |
| `task1_results/ablation/resnet_vs_vit_comparison.json` | Ablation results (ViT Full=98.45%, ResNet=72.39%) |
| `task1_results/kfold_cv_results.md` | Protocol B 3-fold CV summary |
| `task1_results/data_cleaning_impact_report.md` | 4320→3546 cleaning, 2913/316/317 split |
| `task1_scripts/train_vit.py` | Protocol A training script (produces `classifier_best.pt`) |
| `task1_scripts/evaluate_vit_model.py` | Protocol A heldout evaluation |
| `task1_scripts/train_kfold.py` | Protocol B 3-fold CV (resnet50 / vit_b_16 / convnext_tiny) |

### Task 2

| File | Role |
|---|---|
| `gold_standard/test_split.json` | Gold test set labels (ground truth, n=134) |
| `gold_standard/val_split.json` | Gold val set labels (n=100) |
| `lora_exports/qwen2_5_vl_7b/gold_test_predictions_ckpt675.json` | ★ Final benchmark predictions + metrics |
| `task2_vit_baseline/vit_test_metrics.json` | ViT-B/16 baseline metrics |
| `resnet50_gold_test_metrics.json` | ResNet50 baseline metrics (transcribed from logs) |
| `qwen_train_high.json` | Silver-label pool (2271 entries, 2199 unique filenames) |
| `task2_vit_baseline/train_split.json` | Deduped 2199-sample training split |

---

## Dependencies

| Component | Package |
|---|---|
| Deep learning | `torch >= 2.0`, `torchvision` |
| Vision backbone (Task 2 baseline) | `timm` |
| LLM fine-tuning | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| Metrics | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn`, `opencv-python` |
| Image hashing (dedup) | `imagehash` |

---

## Citation

If you use this work, please cite the dissertation as:

```bibtex
@thesis{cao2026eda,
  title       = {EDA Provenance Identification and Schematic-Level Function Tagging
                 for Mixed-Source PCB Schematic Repositories},
  author      = {Cao, Jiajun},
  year        = {2026},
  school      = {University of Nottingham Ningbo China},
  type        = {BEng Final Year Project Dissertation},
  department  = {Department of Electrical and Electronic Engineering}
}
```

### References

Key works cited in this dissertation:

<details>
<summary><strong>AI for EDA / Background</strong></summary>

```bibtex
@article{xu2024ainativeeda,
  title         = {The Dawn of AI-Native EDA: Opportunities and Challenges of Large Circuit Models},
  author        = {Chen, Lei and Chen, Yiqi and Chu, Zhufei and Fang, Wenji and Ho, Tsung-Yi and Huang, Ru and Huang, Yu and others},
  year          = {2024},
  journal       = {arXiv preprint arXiv:2403.07257},
  eprint        = {2403.07257},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR}
}

@article{llm4eda2024,
  title         = {LLM4EDA: Emerging Progress in Large Language Models for Electronic Design Automation},
  author        = {Zhong, Ruizhe and Du, Xingbo and Kai, Shixiong and Tang, Zhentao and Xu, Siyuan and Zhen, Hui-Ling and Hao, Jianye and Xu, Qiang and Yuan, Mingxuan and Yan, Junchi},
  year          = {2024},
  journal       = {arXiv preprint arXiv:2401.12224},
  eprint        = {2401.12224},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR}
}

@article{pan2025llmedasurvey,
  title   = {A Survey of Research in Large Language Models for Electronic Design Automation},
  author  = {Pan, Jingyu and Zhou, Guanglei and Chang, Chen-Chia and Jacobson, Isaac and Hu, Jiang and Chen, Yiran},
  year    = {2025},
  journal = {ACM Transactions on Design Automation of Electronic Systems},
  doi     = {10.1145/3715324}
}

@article{nsf2026aieda,
  title         = {Report for NSF Workshop on AI for Electronic Design Automation},
  author        = {Chen, Deming and Ganesh, Vijay and Li, Weikai and Lin, Yingyan Celine and Liu, Yong and Mitra, Subhasish and Pan, David Z. and Puri, Ruchir and Cong, Jason and Sun, Yizhou},
  year          = {2026},
  journal       = {arXiv preprint arXiv:2601.14541},
  eprint        = {2601.14541},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

</details>

<details>
<summary><strong>Document & Visual Language Models</strong></summary>

```bibtex
@inproceedings{xu2019layoutlm,
  title     = {LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
  author    = {Xu, Yiheng and Li, Minghao and Cui, Lei and Huang, Shaohan and Wei, Furu and Zhou, Ming},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year      = {2020},
  pages     = {1192--1200},
  doi       = {10.1145/3394486.3403172}
}

@inproceedings{kim2021donut,
  title     = {OCR-Free Document Understanding Transformer},
  author    = {Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, Jeongyeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  pages     = {498--517},
  doi       = {10.1007/978-3-031-19815-1_29}
}

@inproceedings{lee2023pix2struct,
  title     = {Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding},
  author    = {Lee, Kenton and Joshi, Mandar and Turc, Iulia and Hu, Hexiang and Liu, Fangyu and Eisenschlos, Julian and Khandelwal, Urvashi and Shaw, Peter and Chang, Ming-Wei and Toutanova, Kristina},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023},
  pages     = {20493--20508}
}

@inproceedings{wang2024docllm,
  title     = {DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding},
  author    = {Wang, Dongsheng and Raman, Natraj and Sibue, Mathieu and Ma, Zhiqiang and Babkin, Petr and Kaur, Simerjot and Pei, Yulong and Nourbakhsh, Armineh and Liu, Xiaomo},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2024},
  pages     = {8554--8570},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{luo2024layoutllm,
  title     = {LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding},
  author    = {Luo, Chuwei and Shen, Yufan and Zhu, Zhaoqing and Zheng, Qi and Yu, Zhi and Yao, Cong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  pages     = {15604--15614}
}

@inproceedings{nacson2025docvlm,
  title     = {DocVLM: Make Your VLM an Efficient Reader},
  author    = {Nacson, Mor Shpigel and Aberdam, Aviad and Ganz, Roy and Ben Avraham, Elad and Golts, Alona and Kittenplon, Yair and Mazor, Shai and Litman, Ron},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  pages     = {29005--29015}
}
```

</details>

<details>
<summary><strong>Vision Backbone Architectures</strong></summary>

```bibtex
@inproceedings{he2016resnet,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016},
  pages     = {770--778},
  doi       = {10.1109/CVPR.2016.90}
}

@inproceedings{dosovitskiy2021vit,
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author    = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{liu2022convnext,
  title     = {A ConvNet for the 2020s},
  author    = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

</details>

<details>
<summary><strong>Parameter-Efficient Fine-Tuning & Training Techniques</strong></summary>

```bibtex
@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}

@inproceedings{szegedy2016rethinking,
  title     = {Rethinking the Inception Architecture for Computer Vision},
  author    = {Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jonathon and Wojna, Zbigniew},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@inproceedings{zhang2018mixup,
  title     = {mixup: Beyond Empirical Risk Minimization},
  author    = {Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018}
}

@inproceedings{yun2019cutmix,
  title     = {CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features},
  author    = {Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}

@inproceedings{loshchilov2019adamw,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}

@inproceedings{loshchilov2017sgdr,
  title     = {{SGDR}: Stochastic Gradient Descent with Warm Restarts},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017}
}

@article{hinton2015kd,
  title   = {Distilling the Knowledge in a Neural Network},
  author  = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal = {arXiv preprint arXiv:1503.02531},
  year    = {2015}
}

@inproceedings{xie2020noisystudent,
  title     = {Self-Training with Noisy Student Improves ImageNet Classification},
  author    = {Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

</details>

<details>
<summary><strong>Data Integrity & Evaluation Methodology</strong></summary>

```bibtex
@article{geirhos2020shortcut,
  title   = {Shortcut Learning in Deep Neural Networks},
  author  = {Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A.},
  journal = {Nature Machine Intelligence},
  year    = {2020},
  volume  = {2},
  number  = {11},
  pages   = {665--673},
  doi     = {10.1038/s42256-020-00257-z}
}

@article{barz2020cifair,
  title   = {Do We Train on Test Data? Purging CIFAR of Near-Duplicates},
  author  = {Barz, Bj{\"o}rn and Denzler, Joachim},
  journal = {Journal of Imaging},
  year    = {2020},
  volume  = {6},
  number  = {6},
  pages   = {41},
  doi     = {10.3390/jimaging6060041}
}

@article{sasse2025leakage,
  title   = {Overview of Leakage Scenarios in Supervised Machine Learning},
  author  = {Sasse, Leonard and Nicolaisen-Sobesky, Eliana and Dukart, Juergen and Eickhoff, Simon B. and G{\"o}tz, Michael and Hamdan, Sami and Komeyer, Vera and Kulkarni, Abhijit and Lahnakoski, Juha M. and Love, Bradley C. and Raimondo, Federico and Patil, Kaustubh R.},
  journal = {Journal of Big Data},
  year    = {2025},
  volume  = {12},
  number  = {1},
  pages   = {135},
  doi     = {10.1186/s40537-025-01193-8}
}

@inproceedings{kohavi1995cv,
  title     = {A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection},
  author    = {Kohavi, Ron},
  booktitle = {Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {1995}
}

@article{powers2011eval,
  title   = {Evaluation: From Precision, Recall and {F}-Measure to {ROC}, Informedness, Markedness and Correlation},
  author  = {Powers, David M. W.},
  journal = {Journal of Machine Learning Technologies},
  volume  = {2},
  number  = {1},
  pages   = {37--63},
  year    = {2011}
}
```

</details>

<details>
<summary><strong>Interpretability & Visualisation</strong></summary>

```bibtex
@inproceedings{selvaraju2017gradcam,
  title     = {Grad-{CAM}: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017}
}

@inproceedings{zeiler2014visualizing,
  title     = {Visualizing and Understanding Convolutional Networks},
  author    = {Zeiler, Matthew D. and Fergus, Rob},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2014}
}

@inproceedings{abnar2020attentionflow,
  title     = {Quantifying Attention Flow in Transformers},
  author    = {Abnar, Samira and Zuidema, Willem},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2020}
}

@article{maaten2008tsne,
  title   = {Visualizing Data using t-{SNE}},
  author  = {van der Maaten, Laurens and Hinton, Geoffrey},
  journal = {Journal of Machine Learning Research (JMLR)},
  volume  = {9},
  pages   = {2579--2605},
  year    = {2008}
}

@book{jolliffe2002pca,
  title     = {Principal Component Analysis},
  author    = {Jolliffe, I. T.},
  publisher = {Springer},
  edition   = {2},
  year      = {2002}
}
```

</details>

<details>
<summary><strong>OCR & Text Detection (Related Methods)</strong></summary>

```bibtex
@inproceedings{zhou2017east,
  title     = {{EAST}: An Efficient and Accurate Scene Text Detector},
  author    = {Zhou, Xinyu and Yao, Cong and Wen, He and Wang, Yuzhi and Zhou, Shangbang and He, Weiran and Liang, Jiajun},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017}
}

@article{shi2017crnn,
  title   = {An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition},
  author  = {Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  volume  = {39},
  number  = {11},
  pages   = {2298--2304},
  year    = {2017}
}
```

</details>

<details>
<summary><strong>Retrieval-Augmented Generation</strong></summary>

```bibtex
@inproceedings{lewis2020rag,
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  author    = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```

</details>

---

## License

Scripts and result files are released under the **MIT License**.  
Raw schematic images are sourced from public GitHub repositories; see individual dataset documentation for provenance and copyright notes.

---

<div align="center">

*University of Nottingham Ningbo China · Department of Electrical and Electronic Engineering*  
*Final Year Project 2025–2026 · Cao Jiajun (曹佳竣)*

</div>
