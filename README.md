# PCB Schematic Repository Classification — Dissertation Artefacts

This repository contains all artefacts accompanying the dissertation:  
**"EDA Provenance Identification and Schematic-Level Function Tagging for Mixed-Source PCB Schematic Repositories"**

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

## Citation

```
[Dissertation citation placeholder — to be filled after submission]
```

## License

Scripts and result files: MIT License.  
Raw schematic images: sourced from public GitHub repositories; see individual dataset README for provenance and copyright notes.
