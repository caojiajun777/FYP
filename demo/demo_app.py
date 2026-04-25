"""
EDA Schematic Intelligent Analysis Demo
========================================
Task 1: EDA Tool Source Classification  (ViT-B/16, Acc 99.37%)
Task 2: Circuit Function Multi-label Classification (Qwen2.5-VL-7B + LoRA, Macro-F1 0.856)

Usage:
    cd FYP_Submission/demo
    python demo_app.py
Then open http://localhost:7860 in your browser.

Prerequisites:
    - Place ViT-B/16 weights at:  models/classifier_best.pt
    - Place LoRA adapter at:      models/best_adapter_fixed/
    - Base model (Qwen2.5-VL-7B-Instruct) must be cached in HuggingFace hub.
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr
import pandas as pd

# ─────────────────────────── 路径配置 ────────────────────────────

BASE_DIR = Path(__file__).parent

# 把 EDA_System 目录加入路径，以便 import lora_demo
EDA_SYSTEM_DIR = BASE_DIR / "EDA_System"
if str(EDA_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(EDA_SYSTEM_DIR))

# Task 1: EDA Tool Classification Model (ViT-B/16)
TASK1_MODEL_PATH = BASE_DIR / "models" / "classifier_best.pt"
TASK1_CLASSES    = ["Altium", "Eagle", "JLC", "KiCad", "OrCAD"]

# Task 2: LoRA fine-tuned Qwen2.5-VL-7B
TASK2_LORA_CHECKPOINT = str(BASE_DIR / "models" / "best_adapter_fixed")
os.environ["LORA_ADAPTER_PATH"] = TASK2_LORA_CHECKPOINT

TASK2_CLASS_DESCRIPTIONS = {
    "power":         "Power Management — PMIC / LDO / Buck / Boost",
    "communication": "Communication — Wi-Fi / BT / Zigbee / RF module",
    "interface":     "Interface — USB / UART / CAN / SPI / I2C",
    "control":       "Control — MCU / FPGA / Motor driver",
    "signal":        "Signal Processing — ADC / DAC / Op-Amp / Filter",
}

# Example images
EXAMPLE_DIR = BASE_DIR / "example_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────── 图像预处理 ──────────────────────────

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────── Task 1 模型 ─────────────────────────

_task1_model = None

def _load_task1():
    global _task1_model
    if _task1_model is not None:
        return _task1_model, None
    if not TASK1_MODEL_PATH.exists():
        return None, f"Model file not found: {TASK1_MODEL_PATH}"
    try:
        model = torchvision.models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(768, len(TASK1_CLASSES))
        cp = torch.load(TASK1_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(cp["model_state_dict"])
        model.to(DEVICE).eval()
        _task1_model = model
        print(f"[Task1] ViT-B/16 loaded | val_acc={cp.get('val_acc', 0):.4f}")
        return _task1_model, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────── Task 2 LoRA 模型 ────────────────────

try:
    from lora_demo import LoRADemoSystem, ADAPTER_PATH as _LORA_PATH
    _lora_system = LoRADemoSystem()
    LORA_IMPORT_OK = True
    print(f"[Task2] lora_demo module loaded | adapter: {_LORA_PATH}")
except Exception as _e:
    _lora_system = None
    LORA_IMPORT_OK = False
    print(f"[Task2] lora_demo import failed: {_e}")

# ─────────────────────────── 启动预加载 ──────────────────────────

print("=" * 60)
print("EDA Schematic Analysis Demo  Initializing...")
print(f"Device: {DEVICE.upper()}")
print("=" * 60)
_load_task1()
print("[Task2] Qwen2.5-VL-7B + LoRA: will load on first click (~1-2 min)")
print("=" * 60)

# ─────────────────────────── 通用工具函数 ────────────────────────

def _pil_from_input(image_input):
    if image_input is None:
        return None
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    return Image.open(image_input).convert("RGB")


def _save_temp(image_input) -> str:
    img = _pil_from_input(image_input)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name

# ─────────────────────────── Task 1 推理 ─────────────────────────

def predict_task1(image_input):
    """返回 (result_text, bar_df, json_str)"""
    if image_input is None:
        return "Please upload an EDA schematic image.", None, ""

    model, err = _load_task1()
    if model is None:
        return f"Model loading failed: {err}", None, ""

    img = _pil_from_input(image_input)
    x = _TRANSFORM(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = sorted(zip(TASK1_CLASSES, probs), key=lambda v: v[1], reverse=True)
    top_name, top_conf = ranked[0]

    lines = [
        f"Predicted Tool : {top_name}",
        f"Confidence     : {top_conf*100:.2f}%",
        f"Inference Time : {elapsed_ms:.1f} ms",
        "",
        "All Class Probabilities:",
    ]
    for name, p in ranked:
        bar = "█" * int(p * 28)
        lines.append(f"  {name:<14} {p*100:5.2f}%  {bar}")
    result_text = "\n".join(lines)

    bar_df = pd.DataFrame({
        "EDA Tool": TASK1_CLASSES,
        "Confidence (%)": [round(p * 100, 2) for p in probs],
    })

    output_json = {
        "task": "Task 1 — EDA Tool Source Classification",
        "model": "ViT-B/16",
        "status": "success",
        "predicted_tool": top_name,
        "confidence": round(top_conf, 4),
        "all_probabilities": {
            name: round(p, 4) for name, p in zip(TASK1_CLASSES, probs)
        },
        "inference_time_ms": round(elapsed_ms, 1),
    }
    return result_text, bar_df, json.dumps(output_json, indent=2, ensure_ascii=False)


# ─────────────────────────── Task 2 推理（LoRA）─────────────────

def predict_task2_lora(image_input):
    """返回 (status_text, json_str)"""
    if image_input is None:
        return "Please upload an EDA schematic image.", ""

    if not LORA_IMPORT_OK or _lora_system is None:
        err_json = json.dumps({"status": "error",
                               "message": "LoRA module failed to load. Check EDA_System/lora_demo.py"},
                              indent=2)
        return "LoRA module failed to load.", err_json

    if not _lora_system.enabled:
        ok = _lora_system.ensure_loaded()
        if not ok:
            err = _lora_system._load_error or "Unknown error"
            err_json = json.dumps({"status": "error", "message": err}, indent=2)
            return f"Model loading failed: {err}", err_json

    tmp_path = _save_temp(image_input)
    try:
        result = _lora_system.analyze(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if result["status"] != "success":
        err_json = json.dumps({"status": "error",
                               "message": result.get("error", "unknown")}, indent=2)
        return f"Inference failed: {result.get('error', 'unknown')}", err_json

    labels     = result["labels"]
    raw_output = result.get("raw_output", "")
    elapsed_s  = result.get("inference_time", 0.0)

    if labels:
        labels_str = " | ".join(f"[{lb}]" for lb in labels)
    else:
        labels_str = "(No labels detected, see raw_model_output in JSON)"
    status_text = (
        f"Predicted Labels : {labels_str}\n"
        f"Raw Output       : {raw_output}\n"
        f"Inference Time   : {elapsed_s*1000:.0f} ms"
    )

    output_json = {
        "task": "Task 2 — EDA Circuit Function Classification",
        "model": "Qwen2.5-VL-7B-Instruct + LoRA (r=16, alpha=32)",
        "checkpoint": "checkpoint-675",
        "status": "success",
        "predicted_labels": labels,
        "label_descriptions": {
            lb: TASK2_CLASS_DESCRIPTIONS.get(lb, lb) for lb in labels
        },
        "raw_model_output": raw_output,
        "inference_time_s": round(elapsed_s, 3),
    }
    return status_text, json.dumps(output_json, indent=2, ensure_ascii=False)


# ─────────────────────────── 综合推理 ────────────────────────────

def predict_combined(image_input):
    if image_input is None:
        return "Please upload an image.", ""

    t1_text, _, t1_json_str = predict_task1(image_input)
    t2_text, t2_json_str    = predict_task2_lora(image_input)

    try:
        t1_obj = json.loads(t1_json_str)
    except Exception:
        t1_obj = {"raw": t1_json_str}
    try:
        t2_obj = json.loads(t2_json_str)
    except Exception:
        t2_obj = {"raw": t2_json_str}

    combined = {
        "task1_eda_tool_classification": t1_obj,
        "task2_function_classification": t2_obj,
    }
    summary = (
        "[Task 1] " + t1_text.split("\n")[0] + "\n" +
        "[Task 2] " + t2_text.split("\n")[0]
    )
    return summary, json.dumps(combined, indent=2, ensure_ascii=False)


# ─────────────────────────── 示例图像列表 ────────────────────────

def _get_examples():
    exs = []
    if EXAMPLE_DIR.exists():
        for p in sorted(EXAMPLE_DIR.glob("*.png"))[:3]:
            exs.append([str(p)])
    return exs or None


# ─────────────────────────── Gradio UI ────────────────────────────

_CSS = """
body { font-family: 'Segoe UI', sans-serif; }
.mono textarea, .mono pre {
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
}
footer { display: none !important; }
"""

_TITLE = """
<div style="text-align:center; padding:10px 0 4px;">
  <h1 style="margin:0; color:#1a5276; font-size:1.7rem;">
    EDA Schematic Intelligent Analysis Demo
  </h1>
  <p style="color:#555; margin:6px 0 0; font-size:0.95rem;">
    Task 1: EDA Tool Source Classification (ViT-B/16)&nbsp;|&nbsp;
    Task 2: Circuit Function Multi-label Classification (Qwen2.5-VL-7B + LoRA)
  </p>
</div>
"""

_TASK1_MD = """
**Task Description**: Identify which EDA tool software the schematic was created with (single-label, 5 classes).

| Model | Classes | Test Accuracy | Macro F1 |
|------|------|------------|--------|
| ViT-B/16 | 5 | **99.05%** | **99.15%** |

Target Classes: `Altium Designer` · `Eagle` · `JLC EDA` · `KiCad` · `OrCAD`
"""

_TASK2_MD = """
**Task Description**: Predict the functional categories of a circuit board (multi-label). The model outputs structured results.

| Model | Classes | Exact Match | Micro-F1 | Macro-F1 |
|------|------|------------|---------|--------|
| Qwen2.5-VL-7B + LoRA | 5 | **56.7%** | **85.7%** | **85.6%** |

Target Categories: `power` · `communication` · `interface` · `control` · `signal`

> **The 7B model will be loaded on first click (~1-2 min). Please wait patiently.**
"""

_COMBINED_MD = """
**Combined Analysis**: Upload one image to run both Task 1 (Tool Classification) and Task 2 (Function Classification).
Results are merged into a single structured JSON output for easy viewing and export.
"""


def build_task1_tab():
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload Schematic", type="pil", height=360)
            btn = gr.Button("Classify", variant="primary", size="lg")
            exs = _get_examples()
            if exs:
                gr.Examples(examples=exs, inputs=inp, label="Example Images")

        with gr.Column(scale=1):
            out_text = gr.Textbox(
                label="Classification Summary",
                lines=9, max_lines=12,
                elem_classes="mono",
            )
            out_bar = gr.BarPlot(
                x="EDA Tool", y="Confidence (%)",
                title="Prediction Probabilities",
                height=230,
                tooltip=["EDA Tool", "Confidence (%)"],
            )
            out_json = gr.Code(
                label="Structured JSON Output",
                language="json",
                lines=14,
            )

    def _run(img):
        return predict_task1(img)

    btn.click(fn=_run, inputs=inp, outputs=[out_text, out_bar, out_json])
    inp.change(fn=_run, inputs=inp, outputs=[out_text, out_bar, out_json])


def build_task2_tab():
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload Schematic", type="pil", height=360)
            btn = gr.Button("Classify", variant="primary", size="lg")
            exs = _get_examples()
            if exs:
                gr.Examples(examples=exs, inputs=inp, label="Example Images")
            gr.Markdown(
                "<small style='color:#e74c3c;'>First run requires loading the 7B model. Please wait ~1-2 min.</small>"
            )

        with gr.Column(scale=1):
            out_text = gr.Textbox(
                label="Classification Summary",
                lines=5, max_lines=7,
                elem_classes="mono",
            )
            out_json = gr.Code(
                label="Structured JSON Output (Full Inference Result)",
                language="json",
                lines=24,
            )

    def _run(img):
        return predict_task2_lora(img)

    btn.click(fn=_run, inputs=inp, outputs=[out_text, out_json])
    inp.change(fn=_run, inputs=inp, outputs=[out_text, out_json])


def build_combined_tab():
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Upload Schematic", type="pil", height=360)
            btn = gr.Button("Run Combined Analysis", variant="primary", size="lg")
            exs = _get_examples()
            if exs:
                gr.Examples(examples=exs, inputs=inp, label="Example Images")

        with gr.Column(scale=1):
            out_summary = gr.Textbox(
                label="Analysis Summary",
                lines=4, max_lines=6,
                elem_classes="mono",
            )
            out_json = gr.Code(
                label="Merged JSON (Task 1 + Task 2)",
                language="json",
                lines=38,
            )

    def _run(img):
        return predict_combined(img)

    btn.click(fn=_run, inputs=inp, outputs=[out_summary, out_json])
    inp.change(fn=_run, inputs=inp, outputs=[out_summary, out_json])


# ─────────────────────────── 主入口 ──────────────────────────────

def main():
    with gr.Blocks(title="EDA Schematic Analysis Demo") as demo:
        gr.HTML(_TITLE)

        with gr.Tabs():
            with gr.TabItem("Task 1: EDA Tool Classification"):
                gr.Markdown(_TASK1_MD)
                build_task1_tab()

            with gr.TabItem("Task 2: Circuit Function Classification (LoRA)"):
                gr.Markdown(_TASK2_MD)
                build_task2_tab()

            with gr.TabItem("Combined Analysis (Task 1 + Task 2)"):
                gr.Markdown(_COMBINED_MD)
                build_combined_tab()

        gr.HTML(
            "<div style='text-align:center;color:#aaa;font-size:11px;margin-top:6px;'>"
            "FYP · EDA Schematic Analysis · ViT-B/16 + Qwen2.5-VL-7B LoRA"
            "</div>"
        )

    import socket
    port = 7860
    for p in range(7860, 7880):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", p)) != 0:
                port = p
                break

    print(f"\nDemo running at: http://localhost:{port}\n")
    demo.launch(server_port=port, inbrowser=True, share=False, css=_CSS)


if __name__ == "__main__":
    main()

