"""
Task 3.1 - ViT特征空间t-SNE可视化

提取ViT-B/16的特征向量，进行降维可视化和类间距离分析
输出: tsne_visualization_vit.png
"""
import os
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# 配置
CLASSES = ["altium", "eagle", "jlc", "kicad", "orcad"]
DATA_ROOT = Path(os.environ.get("TASK1_KFOLD_DATA_ROOT", r"D:\FYP\data\EDA_cls_dataset_kfold"))
OUTPUT_DIR = Path(os.environ.get("TASK1_OUTPUT_DIR", r"D:\FYP\Classifier\paper_results\interpretability\feature_visualization"))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

class ViTFeatureExtractor(nn.Module):
    """提取ViT-B/16的全局特征（池化前）"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # torchvision ViT特征提取流程
        with torch.no_grad():
            x = self.model._process_input(x)  # [B, 3, 224, 224]
            n = x.shape[0]
            x = self.model.conv_proj(x)  # [B, 768, 14, 14]
            x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
            cls_token = self.model.encoder.cls_token.expand(n, -1, -1)  # [B, 1, 768]
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
            x = x + self.model.encoder.pos_embedding[:, :x.size(1), :]
            x = self.model.encoder.dropout(x)
            x = self.model.encoder(x)  # [B, 197, 768]
            # 去掉CLS token
            x = x[:, 1:, :]
            # 对所有patch做均值池化
            out = x.mean(dim=1)
            return out

# 加载ViT模型
print("📦 Loading ViT-B/16 (ImageNet预训练)...")
vit = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1)
vit.heads.head = nn.Linear(vit.heads.head.in_features, len(CLASSES))
feature_extractor = ViTFeatureExtractor(vit).to(DEVICE).eval()

# 加载数据
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
dataset = ImageFolder(DATA_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
print(f"  ✅ Loaded {len(dataset)} samples")

# 提取特征
features_list = []
labels_list = []
with torch.no_grad():
    for images, labels in tqdm(loader, desc="Extracting"):
        images = images.to(DEVICE)
        feats = feature_extractor(images)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
features = np.vstack(features_list)
labels = np.concatenate(labels_list)
print(f"  ✅ Extracted features: shape={features.shape}")

# t-SNE可视化
print(f"\n🎨 Running t-SNE (perplexity=30)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, verbose=1)
tsne_features = tsne.fit_transform(features)

fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
markers = ['o', 's', '^', 'D', 'v']
for idx, class_name in enumerate(CLASSES):
    mask = labels == idx
    ax.scatter(tsne_features[mask, 0], tsne_features[mask, 1],
               c=colors[idx], marker=markers[idx],
               label=class_name.upper(), alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
ax.set_xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
ax.set_title('ViT-B/16 Feature Space - t-SNE Visualization', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
output_path = OUTPUT_DIR / "tsne_visualization_vit.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved t-SNE plot to {output_path}")
