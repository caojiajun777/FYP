"""
Task 3.2 - Grad-CAM (ResNet50 / ViT) 统一修正版

关键修复：
1) target class 默认用 pred（避免你之前 class_idx 强行解释导致错位）
2) ViT 不用 ResNet 那套“通道权重GAP”的Grad-CAM，而用更稳的 token级 (grad*act) 贡献：
   token_score = ReLU(sum_c A[t,c] * dY/dA[t,c])  -> reshape(H,W)
3) 保存 cam_resized.npy (224x224) + selected_samples.json，供 Occlusion 脚本严格对齐
4) 不再使用 register_backward_hook（已弃用），用 retain_grad / tensor hook 获取梯度
"""

import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# =========================
# 配置
# =========================
CLASSES = ["altium", "eagle", "jlc", "kicad", "orcad"]
DATA_ROOT = Path(os.environ.get("TASK1_KFOLD_DATA_ROOT", r"D:\FYP\data\EDA_cls_dataset_kfold"))

MODEL_TYPE = "resnet50"  # 'resnet50' or 'vit'

if MODEL_TYPE == "resnet50":
    MODEL_PATH = Path(os.environ.get("KFOLD_RESNET_MODEL_PATH", r"D:\FYP\runs_kfold\resnet50\fold0\best_model.pt"))
    OUTPUT_DIR = Path(os.environ.get("TASK1_OUTPUT_DIR", r"D:\FYP\Classifier\paper_results\interpretability\grad_cam_resnet50"))
elif MODEL_TYPE == "vit":
    MODEL_PATH = Path(os.environ.get("TASK1_VIT_MODEL_PATH", r"D:\FYP\runs_vit\train_vit_b16_best\classifier_best.pt"))
    OUTPUT_DIR = Path(os.environ.get("TASK1_OUTPUT_DIR", r"D:\FYP\Classifier\paper_results\interpretability\grad_cam_vit"))
else:
    raise ValueError("MODEL_TYPE must be 'resnet50' or 'vit'")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42

TARGET_MODE = "pred"   # 'pred' / 'true' / 'given'
GIVEN_TARGET_CLASS = 0

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# Grad-CAM 实现
# =========================
class GradCAM:
    """
    统一 Grad-CAM：
    - ResNet：标准 Grad-CAM
    - ViT：token级 grad*act 贡献（更稳，不容易全0）
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, model_type: str):
        self.model = model
        self.target_layer = target_layer
        self.model_type = model_type

        self.activations = None  # forward保存
        self.hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # out 可能是tuple，尽量取tensor
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.activations = out
        # 关键：保留梯度（backward后可用 self.activations.grad）
        # 确保 requires_grad=True，然后才能 retain_grad
        if self.activations.requires_grad is False:
            self.activations.requires_grad_(True)
        if hasattr(self.activations, "retain_grad"):
            self.activations.retain_grad()

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        返回：cam_2d (Hc, Wc) for resnet; (14,14) for vit-b/16
        """
        logits = self.model(input_tensor)  # no torch.no_grad
        self.model.zero_grad(set_to_none=True)

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        acts = self.activations
        grads = None if acts is None else acts.grad

        if acts is None or grads is None:
            # 极端情况下hook没抓到，直接返回全0
            return np.zeros((14, 14), dtype=np.float32) if self.model_type == "vit" else np.zeros((7, 7), dtype=np.float32)

        # 统一成 torch.Tensor
        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        if isinstance(grads, (tuple, list)):
            grads = grads[0]

        # -------- ViT：token级贡献 (grad*act) --------
        if self.model_type == "vit":
            # torchvision ViT 常见形状：[B, seq, C]；也有可能是 [seq, B, C]
            if acts.dim() == 3:
                if acts.shape[0] != input_tensor.shape[0] and acts.shape[1] == input_tensor.shape[0]:
                    # [seq, B, C] -> [B, seq, C]
                    acts = acts.permute(1, 0, 2).contiguous()
                    grads = grads.permute(1, 0, 2).contiguous()

                # 去掉CLS token
                acts_no_cls = acts[:, 1:, :]   # [B, T, C]
                grads_no_cls = grads[:, 1:, :] # [B, T, C]

                # token importance: ReLU(sum_c act * grad)
                token_score = (acts_no_cls * grads_no_cls).sum(dim=2)  # [B, T]
                token_score = F.relu(token_score)

                Ttokens = token_score.shape[1]
                side = int(Ttokens ** 0.5)
                if side * side != Ttokens:
                    # 不可reshape就退化成均值
                    cam = token_score[0].detach().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    return cam

                cam = token_score[0].reshape(side, side)  # [H,W]
            else:
                # 兜底
                cam = grads.abs().mean(dim=0)
                cam = cam.detach().cpu().numpy()

        # -------- ResNet：标准 Grad-CAM --------
        else:
            # acts/grads: [B, C, H, W]
            if acts.dim() != 4:
                # 兜底
                cam = grads.abs().mean(dim=0)
                cam = cam.detach().cpu().numpy()
            else:
                weights = grads.mean(dim=(2, 3), keepdim=True)         # [B,C,1,1]
                cam = (weights * acts).sum(dim=1)                      # [B,H,W]
                cam = F.relu(cam)[0]

        cam = cam.detach().cpu().numpy().astype(np.float32)
        # 归一化（注意：如果是常数图会变成全0，这是正常；但至少不会“错误错位”）
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# =========================
# 模型加载
# =========================
def load_model():
    print(f"📦 Loading {MODEL_TYPE.upper()} from {MODEL_PATH} ...")

    if MODEL_TYPE == "resnet50":
        model = tv.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        model = tv.models.vit_b_16(weights=None)
        # torchvision vit head
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
        else:
            # 兜底
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


# =========================
# 采样一致性（导出json给Occlusion用）
# =========================
def select_samples_per_class(dataset, samples_per_class=5, seed=42):
    print(f"\n📋 Selecting {samples_per_class} samples/class (seed={seed})")

    class_indices = {cls: [] for cls in CLASSES}
    for idx, (_, label) in enumerate(dataset):
        class_indices[CLASSES[label]].append(idx)

    random.seed(seed)
    selected = {}
    for cls in CLASSES:
        idxs = class_indices[cls]
        if len(idxs) >= samples_per_class:
            picked = random.sample(idxs, samples_per_class)
        else:
            picked = idxs
        selected[cls] = picked
        print(f"  {cls.upper()}: {len(picked)}")
    return selected


def apply_colormap(cam_2d: np.ndarray, original_img_np: np.ndarray):
    h, w = original_img_np.shape[:2]
    cam_resized = cv2.resize(cam_2d, (w, h), interpolation=cv2.INTER_LINEAR)

    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    alpha = 0.4
    overlay = cv2.addWeighted(original_img_np, 1 - alpha, cam_colored, alpha, 0)
    return overlay, cam_resized


# =========================
def visualize_gradcam(model, dataset, selected_samples):
    print(f"\n🎨 Generating Grad-CAM... -> {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 选 target_layer
    if MODEL_TYPE == "resnet50":
        target_layer = model.layer4[-1]
    else:
        # 更稳的选择：最后一个encoder block里的 LayerNorm（token语义更稳定）
        # torchvision EncoderBlock 通常有 ln_1 / ln_2
        blk = model.encoder.layers[-1]
        target_layer = blk.ln_1 if hasattr(blk, "ln_1") else blk

    grad_cam = GradCAM(model, target_layer, MODEL_TYPE)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    # 保存采样索引，Occlusion脚本直接读这个保证完全一致
    with open(OUTPUT_DIR / "selected_samples.json", "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, indent=2, ensure_ascii=False)

    for class_name in CLASSES:
        print(f"\n  Processing {class_name.upper()}...")
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(exist_ok=True)

        for sample_id, idx in enumerate(selected_samples[class_name], 1):
            img_path = dataset.imgs[idx][0]
            true_label = dataset.imgs[idx][1]

            pil_img = Image.open(img_path).convert("RGB")
            original_img_np = np.array(pil_img.resize((224, 224)))
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            # 预测
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                pred_prob = float(probs[0, pred_idx].item())

            # target class
            if TARGET_MODE == "pred":
                target_idx = pred_idx
            elif TARGET_MODE == "true":
                target_idx = int(true_label)
            elif TARGET_MODE == "given":
                target_idx = int(GIVEN_TARGET_CLASS)
            else:
                raise ValueError("TARGET_MODE must be 'pred' / 'true' / 'given'.")

            cam_small = grad_cam.generate_cam(input_tensor, target_idx)
            overlay, cam_resized = apply_colormap(cam_small, original_img_np)

            # 保存npy（Occlusion对齐就用这个）
            np.save(class_dir / f"sample_{sample_id}_cam.npy", cam_resized)

            # 可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_img_np)
            axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(cam_resized, cmap="jet")
            axes[1].set_title(f"Grad-CAM Heatmap\nTarget: {CLASSES[target_idx].upper()}", fontsize=12, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Grad-CAM Overlay", fontsize=12, fontweight="bold")
            axes[2].axis("off")

            title = (
                f"{class_name.upper()} - Sample {sample_id}\n"
                f"True: {CLASSES[true_label].upper()} | Pred: {CLASSES[pred_idx].upper()} ({pred_prob:.2%})"
            )
            plt.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
            plt.tight_layout()

            out_png = class_dir / f"sample_{sample_id}.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()

            # 同时存一份meta，后面写报告方便
            meta = {
                "img_path": img_path,
                "true_label": int(true_label),
                "pred_label": int(pred_idx),
                "pred_prob": float(pred_prob),
                "target_label": int(target_idx),
                "target_mode": TARGET_MODE,
            }
            with open(class_dir / f"sample_{sample_id}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(f"    ✅ sample_{sample_id}.png + sample_{sample_id}_cam.npy")

    grad_cam.remove()
    print("\n✅ Grad-CAM done.")


def main():
    print("\n" + "=" * 70)
    print(f"🔥 Grad-CAM Generate ({MODEL_TYPE})")
    print("=" * 70)

    model = load_model()
    print(f"\n📂 Loading dataset: {DATA_ROOT}")
    dataset = ImageFolder(DATA_ROOT)
    print(f"  ✅ samples: {len(dataset)}")

    selected_samples = select_samples_per_class(
        dataset,
        samples_per_class=SAMPLES_PER_CLASS,
        seed=RANDOM_SEED,
    )

    visualize_gradcam(model, dataset, selected_samples)

    print("\n" + "=" * 70)
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
