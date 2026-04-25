import json
import os
import torch
import warnings
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings("ignore")

# Set DATA_ROOT to the FYP_repo directory; LORA_ROOT to the LoRA checkpoint parent directory
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LORA_ROOT = os.environ.get("LORA_ROOT", os.path.join(DATA_ROOT, "lora_exports", "qwen2_5_vl_7b"))

TEST_DATA_PATH = os.path.join(DATA_ROOT, "gold_standard", "test_split.json")
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINT = os.path.join(LORA_ROOT, "checkpoint-675")  # use best checkpoint
OUTPUT_JSON = os.path.join(DATA_ROOT, "lora_exports", "qwen2_5_vl_7b", "gold_test_predictions_ckpt675.json")

CATEGORIES = ["power", "interface", "communication", "signal", "control"]

SYSTEM_PROMPT = (
    "You are an expert EDA schematic function classifier. "
    "Analyze the provided schematic image and strictly output the corresponding functional categories. "
    "Rely ONLY on visible circuit evidence (e.g., components, topology, interfaces) and avoid guessing "
    "based solely on background knowledge."
)

USER_TEXT_PROMPT = (
    "Analyze the schematic diagram and identify its functional categories. "
    "The categories must be chosen from the following list ONLY: "
    "[power, interface, communication, signal, control]. "
    "Output the identified categories separated by commas, with no other text."
)

def fix_image_path(old_path: str) -> str:
    """Resolve an image path from the stored JSON to an absolute path on the current machine.
    Set the IMAGE_ROOT environment variable to the directory containing EDA_cls_dataset_full/."""
    image_root = os.environ.get("IMAGE_ROOT", DATA_ROOT)
    normalized = old_path.replace("\\\\", "/").replace("\\", "/")
    if "EDA_cls_dataset_full/" in normalized:
        rel_path = normalized.split("EDA_cls_dataset_full/")[1]
        return os.path.join(image_root, "EDA_cls_dataset_full", rel_path)
    return os.path.join(image_root, "EDA_cls_dataset_full", "jlc", os.path.basename(normalized))

def evaluate():
    from qwen_vl_utils import process_vision_info

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\\n============================")
    print(f"开始测试独立金标 Test Set (共 {len(data)} 张)")
    print(f"============================")

    print("Loading Base Model Qwen2.5-VL-7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading Checkpoint: {CHECKPOINT}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT)
    model.eval()

    # 尽量和训练/前面评测一致，避免 fast processor 差异
    processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=False)

    # 去掉 generate 时 temperature warning
    try:
        model.generation_config.temperature = None
    except Exception:
        pass
    try:
        model.generation_config.do_sample = False
    except Exception:
        pass

    y_true = []
    y_pred = []
    format_errors = 0
    missing_images = 0
    records = []

    for item in tqdm(data):
        img_path = fix_image_path(item["image_path"])
        if not os.path.exists(img_path):
            missing_images += 1
            print(f"[Missing Image] {img_path}")
            continue

        gt_labels = [x.strip().lower() for x in item["gold_labels"] if x.strip()]
        y_true.append(gt_labels)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": USER_TEXT_PROMPT}
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

        pred_labels_raw = [x.strip() for x in output_text.split(",") if x.strip()]
        pred_labels = [x for x in pred_labels_raw if x in CATEGORIES]

        if len(pred_labels_raw) != len(pred_labels) or not output_text:
            format_errors += 1

        y_pred.append(pred_labels)

        records.append({
            "filename": item["filename"],
            "image_path_fixed": img_path,
            "gold_labels": gt_labels,
            "pred_text": output_text,
            "pred_labels": pred_labels,
            "confidence": item.get("confidence", ""),
            "evidence": item.get("evidence", [])
        })

    if missing_images > 0:
        print(f"\\n[WARNING] Missing images: {missing_images}")
    if len(y_true) == 0:
        raise RuntimeError("没有可评测样本，可能图片路径全部不对。")

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    mlb.fit([CATEGORIES])
    yt_bin = mlb.transform(y_true)
    yp_bin = mlb.transform(y_pred)

    macro_f1 = f1_score(yt_bin, yp_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(yt_bin, yp_bin, average="micro", zero_division=0)
    exact_match = accuracy_score(yt_bin, yp_bin)

    print("\\n================ RESULTS ================")
    print(f"Evaluated Samples      : {len(y_true)}")
    print(f"Format Stability Errors: {format_errors} / {len(y_true)}")
    print(f"Exact Match            : {exact_match:.4f}")
    print(f"Micro-F1               : {micro_f1:.4f}")
    print(f"Macro-F1               : {macro_f1:.4f}")

    p_class = precision_score(yt_bin, yp_bin, average=None, zero_division=0)
    r_class = recall_score(yt_bin, yp_bin, average=None, zero_division=0)
    f1_class = f1_score(yt_bin, yp_bin, average=None, zero_division=0)
    support_class = yt_bin.sum(axis=0)

    print("\\nPer-Category Metrics:")
    for i, cat in enumerate(CATEGORIES):
        print(
            f" - {cat:<14}: "
            f"P = {p_class[i]:.4f} | "
            f"R = {r_class[i]:.4f} | "
            f"F1 = {f1_class[i]:.4f} | "
            f"Support = {int(support_class[i])}"
        )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": CHECKPOINT,
            "test_data_path": TEST_DATA_PATH,
            "num_evaluated": len(y_true),
            "format_errors": format_errors,
            "exact_match": exact_match,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_category": {
                cat: {
                    "precision": float(p_class[i]),
                    "recall": float(r_class[i]),
                    "f1": float(f1_class[i]),
                    "support": int(support_class[i]),
                }
                for i, cat in enumerate(CATEGORIES)
            },
            "predictions": records
        }, f, ensure_ascii=False, indent=2)

    print(f"\\nSaved predictions to: {OUTPUT_JSON}")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("需要安装 qwen-vl-utils，请先运行: pip install qwen-vl-utils")
        raise SystemExit(1)

    evaluate()
