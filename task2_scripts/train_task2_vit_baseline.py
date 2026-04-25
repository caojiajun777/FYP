import os
import json
import random
from typing import List

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# =========================
# Config
# =========================
# Set DATA_ROOT to the FYP_repo directory.
# e.g.  export DATA_ROOT=/path/to/FYP_repo   (Linux/Mac)
#        $env:DATA_ROOT="D:\FYP\...\FYP_repo"  (Windows PowerShell)
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_JSON = os.path.join(DATA_ROOT, "task2_vit_baseline", "train_split.json")
VAL_JSON   = os.path.join(DATA_ROOT, "gold_standard", "val_split.json")
TEST_JSON  = os.path.join(DATA_ROOT, "gold_standard", "test_split.json")

SAVE_DIR = os.path.join(DATA_ROOT, "task2_vit_baseline")
os.makedirs(SAVE_DIR, exist_ok=True)

CATEGORIES = ["power", "interface", "communication", "signal", "control"]
NUM_CLASSES = len(CATEGORIES)

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.5
SEED = 42
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_image_path(path: str) -> str:
    if os.path.exists(path):
        return path

    normalized = path.replace("\\\\", "/").replace("\\", "/")
    image_root = os.environ.get("IMAGE_ROOT", DATA_ROOT)
    if "EDA_cls_dataset_full/" in normalized:
        rel_path = normalized.split("EDA_cls_dataset_full/")[1]
        return os.path.join(image_root, "EDA_cls_dataset_full", rel_path)

    return os.path.join(image_root, "EDA_cls_dataset_full", "jlc", os.path.basename(normalized))

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def labels_to_multihot(labels: List[str]) -> np.ndarray:
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for x in labels:
        if x in CATEGORIES:
            vec[CATEGORIES.index(x)] = 1.0
    return vec


# =========================
# Dataset
# =========================
class SchematicDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.data = load_json(json_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = resolve_image_path(item["image_path"])
        labels = [x.strip().lower() for x in item["gold_labels"] if x.strip()]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = labels_to_multihot(labels)
        return image, torch.tensor(target, dtype=torch.float32), item["filename"]


# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    exact_match = accuracy_score(y_true, y_pred)

    p_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "exact_match": exact_match,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "p_class": p_class,
        "r_class": r_class,
        "f1_class": f1_class,
        "y_pred": y_pred
    }


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_targets = []
    all_probs = []
    all_files = []

    for images, targets, filenames in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_targets.append(targets.numpy())
        all_probs.append(probs)
        all_files.extend(filenames)

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(y_true, y_prob, threshold)

    return metrics, y_true, y_prob, all_files


def main():
    set_seed(SEED)

    print("=" * 80)
    print("Device:", DEVICE)
    print("Train JSON:", TRAIN_JSON)
    print("Val JSON  :", VAL_JSON)
    print("Test JSON :", TEST_JSON)
    print("=" * 80)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_set = SchematicDataset(TRAIN_JSON, transform=train_tf)
    val_set   = SchematicDataset(VAL_JSON, transform=eval_tf)
    test_set  = SchematicDataset(TEST_JSON, transform=eval_tf)

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples  : {len(val_set)}")
    print(f"Test samples : {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_path = os.path.join(SAVE_DIR, "best_vit_task2.pth")

    train_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        val_metrics, _, _, _ = evaluate(model, val_loader, THRESHOLD)

        record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_exact_match": float(val_metrics["exact_match"]),
            "val_micro_f1": float(val_metrics["micro_f1"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        train_history.append(record)

        print(
            f"[Epoch {epoch+1:02d}/{EPOCHS}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_exact={val_metrics['exact_match']:.4f} "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    with open(os.path.join(SAVE_DIR, "vit_train_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_history, f, ensure_ascii=False, indent=2)

    print("\nLoading best model for final test...")
    print(f"Best epoch: {best_epoch}, best val macro-F1: {best_macro_f1:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    test_metrics, y_true, y_prob, all_files = evaluate(model, test_loader, THRESHOLD)

    print("\n=============================================")
    print("        VIT BASELINE FINAL TEST")
    print("=============================================")
    print(f"Exact Match      : {test_metrics['exact_match']:.4f}")
    print(f"Micro-F1         : {test_metrics['micro_f1']:.4f}")
    print(f"Macro-F1         : {test_metrics['macro_f1']:.4f}")
    print("\n| 类别 | Precision | Recall | F1-Score |")
    print("|---|---:|---:|---:|")
    for i, cat in enumerate(CATEGORIES):
        print(
            f"| `{cat}` | "
            f"{test_metrics['p_class'][i]:.4f} | "
            f"{test_metrics['r_class'][i]:.4f} | "
            f"**{test_metrics['f1_class'][i]:.4f}** |"
        )

    y_pred = test_metrics["y_pred"]
    pred_json = []
    for i, fn in enumerate(all_files):
        gt_labels = [CATEGORIES[j] for j in range(NUM_CLASSES) if y_true[i, j] == 1]
        pd_labels = [CATEGORIES[j] for j in range(NUM_CLASSES) if y_pred[i, j] == 1]
        pred_json.append({
            "filename": fn,
            "gold_labels": gt_labels,
            "pred_labels": pd_labels,
            "probabilities": {CATEGORIES[j]: float(y_prob[i, j]) for j in range(NUM_CLASSES)}
        })

    with open(os.path.join(SAVE_DIR, "vit_test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(pred_json, f, ensure_ascii=False, indent=2)

    with open(os.path.join(SAVE_DIR, "vit_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exact_match": float(test_metrics["exact_match"]),
            "micro_f1": float(test_metrics["micro_f1"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "per_category": {
                CATEGORIES[i]: {
                    "precision": float(test_metrics["p_class"][i]),
                    "recall": float(test_metrics["r_class"][i]),
                    "f1": float(test_metrics["f1_class"][i]),
                } for i in range(NUM_CLASSES)
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
