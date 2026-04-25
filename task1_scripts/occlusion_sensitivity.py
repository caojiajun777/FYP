"""
Task 3.3 - Occlusion Sensitivity（严格对齐版）

关键修复：
1) 不再从Grad-CAM png裁剪，直接读取 Grad-CAM 脚本保存的 sample_X_cam.npy（224x224）
2) target class 默认使用 pred（否则会出现“对不上”的错觉）
3) 遮挡值使用“归一化后的灰色”，而不是直接在normalized tensor里写0.5
4) 读取 selected_samples.json，确保与Grad-CAM使用同一批样本
"""

import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

# =========================
# 配置
# =========================
CLASSES = ["altium", "eagle", "jlc", "kicad", "orcad"]
DATA_ROOT = Path(os.environ.get("TASK1_KFOLD_DATA_ROOT", r"D:\FYP\data\EDA_cls_dataset_kfold"))

MODEL_TYPE = "vit"  # Occlusion一般对CNN更直观；你也可以改成 'vit'
if MODEL_TYPE == "resnet50":
    MODEL_PATH = Path(os.environ.get("KFOLD_RESNET_MODEL_PATH", r"D:\FYP\runs_kfold\resnet50\fold0\best_model.pt"))
elif MODEL_TYPE == "vit":
    MODEL_PATH = Path(os.environ.get("TASK1_VIT_MODEL_PATH", r"D:\FYP\runs_vit\train_vit_b16_best\classifier_best.pt"))
else:
    raise ValueError("MODEL_TYPE must be 'resnet50' or 'vit'")

# 对齐：这里指向 Grad-CAM 输出目录（读取 selected_samples.json + cam.npy）
GRADCAM_DIR = Path(os.environ.get("TASK1_GRADCAM_DIR",
    r"D:\FYP\Classifier\paper_results\interpretability\grad_cam_vit" if MODEL_TYPE == "vit"
    else r"D:\FYP\Classifier\paper_results\interpretability\grad_cam_resnet50"))

OUTPUT_DIR = Path(os.environ.get("TASK1_OUTPUT_DIR",
    r"D:\FYP\Classifier\paper_results\interpretability\occlusion_sensitivity_aligned")) / MODEL_TYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# 遮挡参数
PATCH_SIZE = 28
STRIDE = 14

# 关键：遮挡颜色（RGB空间的灰色0.5） -> 转成 Normalize 后的值
OCCLUSION_RGB = [0.5, 0.5, 0.5]

TARGET_MODE = "pred"  # 'pred' / 'true' / 'given'
GIVEN_TARGET_CLASS = 0

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
def load_model():
    print(f"📦 Loading {MODEL_TYPE.upper()} from {MODEL_PATH} ...")

    if MODEL_TYPE == "resnet50":
        model = tv.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        model = tv.models.vit_b_16(weights=None)
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
        else:
            model.head = nn.Linear(model.head.in_features, len(CLASSES))

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    print("  ✅ Model loaded.")
    return model


def get_occlusion_fill_value():
    # gray(0.5) -> normalized: (x-mean)/std for each channel
    fill = []
    for m, s, v in zip(MEAN, STD, OCCLUSION_RGB):
        fill.append((v - m) / s)
    return torch.tensor(fill, dtype=torch.float32).view(3, 1, 1).to(DEVICE)


def occlude_image(img_tensor, x, y, patch_size, fill_value_chw):
    occluded = img_tensor.clone()
    occluded[:, :, y:y+patch_size, x:x+patch_size] = fill_value_chw
    return occluded


def compute_occlusion_sensitivity(model, img_tensor, target_class, patch_size=28, stride=14):
    _, _, h, w = img_tensor.shape

    with torch.no_grad():
        original_logits = model(img_tensor)
        original_prob = torch.softmax(original_logits, dim=1)[0, target_class].item()

    sensitivity_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    fill_value = get_occlusion_fill_value()

    positions = [(x, y) for y in range(0, h - patch_size + 1, stride)
                        for x in range(0, w - patch_size + 1, stride)]

    for x, y in tqdm(positions, desc="  Occlusion", leave=False):
        occluded = occlude_image(img_tensor, x, y, patch_size, fill_value)

        with torch.no_grad():
            logits = model(occluded)
            prob = torch.softmax(logits, dim=1)[0, target_class].item()

        drop = original_prob - prob  # 越大越敏感
        sensitivity_map[y:y+patch_size, x:x+patch_size] += drop
        count_map[y:y+patch_size, x:x+patch_size] += 1

    sensitivity_map = np.divide(
        sensitivity_map, count_map,
        where=count_map > 0,
        out=np.zeros_like(sensitivity_map)
    )

    # 归一化到[0,1]便于显示（但统计建议用未归一化版本）
    vis = sensitivity_map.copy()
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)

    return sensitivity_map, vis, original_prob


def load_selected_samples():
    path = GRADCAM_DIR / "selected_samples.json"
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到 {path}，请先运行 Grad-CAM 脚本生成 selected_samples.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gradcam_npy(class_name, sample_id):
    cam_path = GRADCAM_DIR / class_name / f"sample_{sample_id}_cam.npy"
    if cam_path.exists():
        return np.load(cam_path)  # 224x224
    return None


def main():
    print("\n" + "="*70)
    print(f"🔍 Occlusion Sensitivity (Aligned) - {MODEL_TYPE}")
    print("="*70)

    model = load_model()

    print(f"\n📂 Loading dataset: {DATA_ROOT}")
    dataset = ImageFolder(DATA_ROOT)
    print(f"  ✅ samples: {len(dataset)}")

    selected_samples = load_selected_samples()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    for class_name in CLASSES:
        print(f"\n  Processing {class_name.upper()}...")
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(exist_ok=True)

        indices = selected_samples[class_name]
        for sample_id, idx in enumerate(indices, 1):
            img_path = dataset.imgs[int(idx)][0]
            true_label = dataset.imgs[int(idx)][1]

            pil_img = Image.open(img_path).convert("RGB")
            original_img_np = np.array(pil_img.resize((224, 224)))
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            # pred
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                pred_prob = float(probs[0, pred_idx].item())

            # target
            if TARGET_MODE == "pred":
                target_idx = pred_idx
            elif TARGET_MODE == "true":
                target_idx = int(true_label)
            elif TARGET_MODE == "given":
                target_idx = int(GIVEN_TARGET_CLASS)
            else:
                raise ValueError("TARGET_MODE must be 'pred' / 'true' / 'given'.")

            raw_map, vis_map, original_prob = compute_occlusion_sensitivity(
                model, input_tensor, target_idx, PATCH_SIZE, STRIDE
            )

            # load gradcam
            cam = load_gradcam_npy(class_name, sample_id)

            # overlay
            sens_colored = cv2.applyColorMap(np.uint8(255 * vis_map), cv2.COLORMAP_HOT)
            sens_colored = cv2.cvtColor(sens_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original_img_np, 0.5, sens_colored, 0.5, 0)

            if cam is not None:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(original_img_np)
                axes[0].set_title(f"Original\nTrue:{CLASSES[true_label].upper()} | Pred:{CLASSES[pred_idx].upper()}({pred_prob:.2%})", fontweight="bold")
                axes[0].axis("off")

                im1 = axes[1].imshow(vis_map, cmap="hot")
                axes[1].set_title("Occlusion Sensitivity", fontweight="bold")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                axes[2].imshow(cam, cmap="jet")
                axes[2].set_title("Grad-CAM (npy)", fontweight="bold")
                axes[2].axis("off")

                axes[3].imshow(overlay)
                axes[3].set_title("Sensitivity Overlay", fontweight="bold")
                axes[3].axis("off")
            else:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original_img_np)
                axes[0].set_title(f"Original\nTrue:{CLASSES[true_label].upper()} | Pred:{CLASSES[pred_idx].upper()}({pred_prob:.2%})", fontweight="bold")
                axes[0].axis("off")

                im1 = axes[1].imshow(vis_map, cmap="hot")
                axes[1].set_title("Occlusion Sensitivity", fontweight="bold")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                axes[2].imshow(overlay)
                axes[2].set_title("Sensitivity Overlay", fontweight="bold")
                axes[2].axis("off")

            plt.suptitle(f"{class_name.upper()} - Sample {sample_id} | Target:{CLASSES[target_idx].upper()} ({original_prob:.2%})",
                         fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()

            out_png = class_dir / f"sample_{sample_id}.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()

            np.save(class_dir / f"sample_{sample_id}_sensitivity_raw.npy", raw_map)

            print(f"    ✅ {out_png.name} (+ raw.npy)")

    print("\n" + "="*70)
    print(f"✅ Done. Saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
