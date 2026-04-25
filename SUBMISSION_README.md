# FYP Submission — EDA Schematic Intelligent Analysis

> **EDA Provenance Identification and Schematic-Level Function Tagging for Mixed-Source PCB Schematic Repositories**

## Quick Navigation

```
FYP_Submission/
├── README.md                  ← You are here
│
├── demo/                      ← Gradio demo application
│   ├── demo_app.py            ← Main entry: python demo_app.py
│   ├── EDA_System/
│   │   └── lora_demo.py       ← LoRA inference module (Qwen2.5-VL-7B)
│   ├── example_images/        ← 3 sample schematics for demo
│   └── models/                ← ⚠️ Place model weights here (see below)
│
├── task1_scripts/             ← Task 1 training, evaluation & ablation code
│   ├── train_vit.py           ← ViT-B/16 main training script
│   ├── train_baselines.py     ← ResNet-18/34/50 baselines
│   ├── train_kfold.py         ← 5-fold cross-validation
│   ├── clean_dataset.py       ← Data cleaning pipeline
│   ├── evaluate_vit_model.py  ← Test set evaluation
│   ├── grad_cam.py            ← Grad-CAM interpretability
│   ├── occlusion_sensitivity.py
│   ├── ablation_*.py          ← Ablation study scripts
│   └── ...
│
├── task1_results/             ← Task 1 metrics, figures & reports
│   ├── metrics/               ← Test/val JSON metrics
│   ├── visualizations/        ← Confusion matrices, loss curves
│   ├── ablation/              ← Ablation study results
│   └── *.md                   ← Analysis reports
│
├── task2_scripts/             ← Task 2 training & evaluation code
│   ├── train_task2_qwen_vl_lora.py  ← Qwen2.5-VL-7B LoRA fine-tuning
│   ├── evaluate_gold_test.py        ← Gold test set evaluation
│   ├── prepare_lora_dataset.py      ← Dataset preparation
│   └── ...
│
└── task2_results/             ← Task 2 metrics & predictions
    ├── test_metrics.json
    ├── lora_exports/          ← Per-image predictions (ckpt-675, ckpt-400)
    ├── gold_standard/         ← Gold annotation splits
    └── vit_baseline/          ← ViT baseline comparison
```

## Key Results

| Task | Model | Key Metric | Value |
|------|-------|-----------|-------|
| Task 1 | ViT-B/16 | Test Accuracy | **99.05%** |
| Task 1 | ViT-B/16 | Macro-F1 | **99.15%** |
| Task 2 | Qwen2.5-VL-7B + LoRA | Exact Match | **56.72%** |
| Task 2 | Qwen2.5-VL-7B + LoRA | Micro-F1 | **85.66%** |
| Task 2 | Qwen2.5-VL-7B + LoRA | Macro-F1 | **85.56%** |

## Running the Demo

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥16 GB VRAM (for Task 2 inference)
- Dependencies: `pip install torch torchvision transformers peft gradio qwen-vl-utils pillow pandas`

### Model Weights Setup

Create a `demo/models/` directory and place the following files:

1. **Task 1 — ViT-B/16 weights**
   ```
   demo/models/classifier_best.pt
   ```

2. **Task 2 — LoRA adapter** (copy the entire folder)
   ```
   demo/models/best_adapter_fixed/
   ├── adapter_config.json
   ├── adapter_model.safetensors
   └── ...
   ```

3. **Task 2 — Base model** must be cached locally:
   ```
   Qwen/Qwen2.5-VL-7B-Instruct  (via HuggingFace Hub)
   ```

### Launch

```bash
cd FYP_Submission/demo
python demo_app.py
# Open http://localhost:7860
```

## Dependencies

```
torch >= 2.0
torchvision
transformers == 4.49.0
peft >= 0.18
gradio >= 4.0
qwen-vl-utils
pillow
pandas
numpy
```

## Author

**Cao Jiajun** — BEng EEE, University of Nottingham Ningbo China (UNNC), 2025–2026
