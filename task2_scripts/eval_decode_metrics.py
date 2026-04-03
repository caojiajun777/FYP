import json
import os
import torch
import warnings
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings('ignore')

# Set DATA_ROOT to the FYP_repo directory; LORA_ROOT to the directory containing checkpoint-400 / checkpoint-675
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LORA_ROOT = os.environ.get("LORA_ROOT", os.path.join(DATA_ROOT, "lora_exports", "qwen2_5_vl_7b"))

VAL_DATA_PATH = os.path.join(DATA_ROOT, "qwen_train_high.json")
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINTS = [
    os.path.join(LORA_ROOT, "checkpoint-400"),
    os.path.join(LORA_ROOT, "checkpoint-675"),
]

CATEGORIES = ["power", "interface", "communication", "signal", "control"]

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:100]

def evaluate_checkpoint(ckpt_path, data):
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*50}\nEvaluating Checkpoint: {ckpt_path}\n{'='*50}")

    print("Loading Base Model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Applying LoRA Weights from {ckpt_path}...")
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=False)

    y_true = []
    y_pred = []
    format_errors = 0

    print("Running Inference...")
    for idx, item in enumerate(tqdm(data)):
        img_path = item["images"][0]
        sys_prompt = item["messages"][0]["content"]
        user_prompt = item["messages"][1]["content"]
        gt_labels_str = item["messages"][2]["content"].lower().strip()

        gt_labels = [l.strip() for l in gt_labels_str.split(",") if l.strip()]
        y_true.append(gt_labels)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_prompt.replace("<image>", "").strip()}
            ]}
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )

        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, prompt_len:]

        output_text = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip().lower()

        pred_labels_raw = [l.strip() for l in output_text.split(",") if l.strip()]
        pred_labels = [l for l in pred_labels_raw if l in CATEGORIES]

        if len(pred_labels_raw) != len(pred_labels) or not output_text:
            format_errors += 1

        y_pred.append(pred_labels)

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    mlb.fit([CATEGORIES])
    yt_bin = mlb.transform(y_true)
    yp_bin = mlb.transform(y_pred)

    macro_f1 = f1_score(yt_bin, yp_bin, average='macro', zero_division=0)
    micro_f1 = f1_score(yt_bin, yp_bin, average='micro', zero_division=0)
    exact_match = accuracy_score(yt_bin, yp_bin)

    print("\n--- RESULTS ---")
    print(f"Format Stability Errors: {format_errors} / {len(data)}")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nPer-Category Metrics:")

    p_class = precision_score(yt_bin, yp_bin, average=None, zero_division=0)
    r_class = recall_score(yt_bin, yp_bin, average=None, zero_division=0)
    f1_class = f1_score(yt_bin, yp_bin, average=None, zero_division=0)

    for i, cat in enumerate(CATEGORIES):
        print(f" - {cat:<14}: P = {p_class[i]:.4f} | R = {r_class[i]:.4f} | F1 = {f1_class[i]:.4f}")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("需要安装 qwen-vl-utils，请先运行: pip install qwen-vl-utils")
        sys.exit(1)

    data = load_data(VAL_DATA_PATH)
    print(f"Loaded {len(data)} images for evaluation.")

    for ckpt in CHECKPOINTS:
        evaluate_checkpoint(ckpt, data)
