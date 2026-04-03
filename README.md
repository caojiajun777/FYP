# PCB Schematic Repository Classification — Dissertation Artefacts

This repository contains all artefacts accompanying the dissertation:  
**"EDA Provenance Identification and Schematic-Level Function Tagging for Mixed-Source PCB Schematic Repositories"**

---

## Repository Structure

```
FYP_repo/
├── gold_standard/              # Task 2 gold benchmark splits
│   ├── test_split.json         # 134-image gold test set (human-verified labels)
│   ├── val_split.json          # 100-image gold validation set
│   └── gold_val_test.json      # Merged val+test manifest (leakage guard)
│
├── lora_exports/
│   └── qwen2_5_vl_7b/
│       ├── checkpoint-675/                         # Best LoRA checkpoint (tracked via Git LFS)
│       ├── checkpoint-400/                         # Comparison checkpoint
│       ├── gold_test_predictions_ckpt675.json      # Final benchmark: EM=0.5672, Micro-F1=0.8566
│       └── gold_test_predictions_ckpt400.json      # Checkpoint-400 comparison results
│
├── task2_vit_baseline/
│   ├── train_split.json        # 2199-sample overlap-cleaned training pool
│   ├── vit_test_metrics.json   # ViT-B/16 baseline: EM=0.2388, Micro-F1=0.6561
│   ├── vit_test_predictions.json
│   └── best_vit_task2.pth      # ViT model weights (tracked via Git LFS)
│
├── task2_scripts/
│   ├── build_task2_vit_train_split.py    # Builds 2199-sample train split from 2271 candidates
│   ├── train_task2_vit_baseline.py       # ViT-B/16 multi-label training
│   ├── evaluate_gold_test.py             # Evaluates Qwen LoRA on gold test set
│   └── eval_decode_metrics.py            # Checkpoint comparison evaluation
│
├── qwen_train_high.json        # 2271-entry silver-label pool (high-confidence Qwen outputs)
├── train_resnet_baseline.py    # ResNet50 5-label multi-label training script
├── train_task2_qwen_vl_lora.py # Qwen2.5-VL-7B LoRA fine-tuning script
├── prepare_lora_dataset.py     # Builds LLaMA-Factory training JSON from silver labels
├── task2_gold_models_test_split.json   # Cross-model gold test results summary
└── task2_qwen_vl_lora/        # Additional LoRA adapter snapshots
```

> **Note:** `EDA_cls_dataset/` (Task 1 image data) and `EDA_cls_dataset_full/` (Task 2 image data) are tracked via Git LFS. Raw images are large (~3546 and ~2271 files respectively); set `DATA_ROOT` and `IMAGE_ROOT` environment variables when running scripts locally (see below).

---

## Task 1 — EDA Provenance Identification

**5 classes:** Altium · KiCad · OrCAD · Eagle · JLC/EasyEDA  
**Dataset:** 4320 raw → 3546 after deduplication and cleaning  
**Protocol A fixed split:** train 2913 / val 316 / test 317  
**Protocol B:** stratified 3-fold cross-validation

**Final result (ViT-B/16 on 317-image held-out test):**
| Metric | Value |
|---|---|
| Accuracy | **99.05%** |
| Macro-F1 | **99.15%** |

Detailed Task 1 result files are in `paper_results/` (see `task1_source_classification_cleaned/metrics/`).

---

## Task 2 — Schematic-Level Function Tagging

**5 labels:** power · interface · communication · signal · control  
**Silver-label training pool:** 2271 candidates → 2199 unique (72 internal duplicates removed)  
**Gold benchmark:** 100-image val split + 134-image test split (human-verified)

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

---

## Reproducing the Task 2 Experiments

### 1. Set environment variables

```bash
# Linux / Mac
export DATA_ROOT=/path/to/FYP_repo
export IMAGE_ROOT=/path/to/directory_containing_EDA_cls_dataset_full

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

| File | Role |
|---|---|
| `gold_standard/test_split.json` | Gold test set labels (ground truth) |
| `gold_standard/val_split.json` | Gold val set labels |  
| `lora_exports/qwen2_5_vl_7b/gold_test_predictions_ckpt675.json` | Final benchmark predictions + metrics |
| `task2_vit_baseline/vit_test_metrics.json` | ViT-B/16 baseline metrics |
| `qwen_train_high.json` | Silver-label training pool (2271 entries) |
| `task2_vit_baseline/train_split.json` | Deduped 2199-sample training split |
| `task2_gold_models_test_split.json` | Cross-model comparison on gold test |

---

## Citation

```
[Dissertation citation placeholder — to be filled after submission]
```

## License

Scripts and result files: MIT License.  
Raw schematic images: sourced from public GitHub repositories; see individual dataset README for provenance and copyright notes.
