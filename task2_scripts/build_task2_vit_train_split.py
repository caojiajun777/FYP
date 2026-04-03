import json
import os
from collections import Counter

# Set DATA_ROOT to the directory containing qwen_train_high.json and gold_standard/
# e.g.  export DATA_ROOT=/path/to/FYP_repo   (Linux/Mac)
#        $env:DATA_ROOT="D:\FYP\Classifier\paper_results\FYP_repo"  (Windows PowerShell)
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QWEN_TRAIN_JSON = os.path.join(DATA_ROOT, "qwen_train_high.json")
VAL_JSON = os.path.join(DATA_ROOT, "gold_standard", "val_split.json")
TEST_JSON = os.path.join(DATA_ROOT, "gold_standard", "test_split.json")

OUT_DIR = os.path.join(DATA_ROOT, "task2_vit_baseline")
OUT_TRAIN = os.path.join(OUT_DIR, "train_split.json")

CATEGORIES = ["power", "interface", "communication", "signal", "control"]

os.makedirs(OUT_DIR, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_labels(text):
    labels = [x.strip().lower() for x in text.split(",") if x.strip()]
    labels = [x for x in labels if x in CATEGORIES]
    seen = set()
    uniq = []
    for x in labels:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def main():
    qwen_data = load_json(QWEN_TRAIN_JSON)
    val_data = load_json(VAL_JSON)
    test_data = load_json(TEST_JSON)

    forbidden = set(os.path.basename(x["image_path"].replace("\\", "/")) for x in val_data)
    forbidden |= set(os.path.basename(x["image_path"].replace("\\", "/")) for x in test_data)

    train_samples = []
    used = set()
    label_counter = Counter()
    skipped_overlap = 0
    skipped_empty = 0

    for item in qwen_data:
        img_path = item["images"][0]
        filename = os.path.basename(img_path)

        if filename in forbidden:
            skipped_overlap += 1
            continue
        if filename in used:
            continue

        assistant_text = None
        for msg in item["messages"]:
            if msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
                break

        if assistant_text is None:
            skipped_empty += 1
            continue

        labels = parse_labels(assistant_text)
        if not labels:
            skipped_empty += 1
            continue

        train_samples.append({
            "filename": filename,
            "image_path": img_path,
            "gold_labels": labels,
            "source": "qwen_train_high"
        })
        used.add(filename)

        for lb in labels:
            label_counter[lb] += 1

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Saved train split to:", OUT_TRAIN)
    print("Original qwen_train_high size :", len(qwen_data))
    print("Skipped due to val/test overlap:", skipped_overlap)
    print("Skipped due to empty labels    :", skipped_empty)
    print("Final train samples            :", len(train_samples))
    print("Val samples                    :", len(val_data))
    print("Test samples                   :", len(test_data))
    print("Label counts                   :", dict(label_counter))
    print("=" * 80)

if __name__ == "__main__":
    main()
