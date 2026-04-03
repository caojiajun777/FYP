"""
使用 Vision Transformer (ViT) 训练 EDA 分类器
相比 ResNet，ViT 在视觉分类任务上可能有更好的性能
"""

import os, sys, math, time, json, random
from pathlib import Path
from collections import Counter
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============== 配置 ==============

CONFIG = {
    "data_root": os.environ.get("TASK1_DATA_ROOT", r"D:\FYP\data\EDA_cls_dataset"),

    # 训练参数
    "epochs": 50,          # 实际上 20~30 就够，你反正有 early stopping
    "batch_size": 32,

    # 学习率相关
    "lr": 1.5e-4,          # 从 3e-4 降一半，更适合小数据 fine-tune
    "warmup_epochs": 3,    # 3~5 就够了，别拖到 10
    "weight_decay": 0.05,  # 可以先保持

    "img_size": 224,
    "gamma": 2.0,
    "patience": 10,        # 早停更敏感一点，避免在高 LR 区域瞎晃太久
    "seed": 42,
    "num_workers": 4,

    "model_type": "vit_b_16",

    # 特征保留
    "use_grayscale": False,
    "mask_footer": False,

    # 数据增强
    "use_augmentation": True,
    "mixup_alpha": 0.2,    # 保留 Mixup
    "cutmix_alpha": 0.0,   # 先关掉 CutMix，等曲线稳定再考虑打开
    "label_smoothing": 0.05,  # 略减轻一点平滑强度

    "output_dir": os.environ.get("TASK1_VIT_OUTPUT_DIR", r"D:\FYP\runs_vit\train_vit_b16_best"),
}


CLASSES = ["altium", "kicad", "orcad", "eagle", "jlc"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ============== 工具函数 ==============

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def focal_loss(logits, target, gamma=2.0, weight=None):
    """Focal Loss with optional class weights"""
    ce = F.cross_entropy(logits, target, reduction='none', weight=weight)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def macro_f1(y_true, y_pred, num_classes):
    """Pure PyTorch Macro F1"""
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def plot_confusion(cm, class_names, out_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (ViT)', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def load_config(cfg):
    """转换配置为 Namespace"""
    return types.SimpleNamespace(**cfg)


# ============== 数据集 ==============

class ViTDataset(ImageFolder):
    """ViT 数据集（支持增强）"""
    
    def __init__(self, root, transform, use_grayscale=False):
        super().__init__(root)
        self.transform_fn = transform
        self.use_grayscale = use_grayscale

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        if self.use_grayscale:
            img_gray = img.convert('L')
            img = Image.merge('RGB', (img_gray, img_gray, img_gray))
        
        if self.transform_fn:
            img = self.transform_fn(img)
        
        return img, target


# ============== 主程序 ==============

def main():
    args = load_config(CONFIG)
    
    # 初始化
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("🚀 Vision Transformer (ViT) EDA 分类器训练")
    print("="*70)
    print(f"🔧 设备: {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔧 模型: {args.model_type.upper()}")
    print(f"🔧 图像尺寸: {args.img_size}")
    print(f"🔧 Batch Size: {args.batch_size}")
    print(f"🔧 数据增强: {'启用' if args.use_augmentation else '禁用'}")
    print("="*70 + "\n")
    
    # 路径检查
    root = Path(args.data_root)
    train_dir = root / 'train'
    val_dir = root / 'val_cropped'
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ 目录不存在:\n  train: {train_dir}\n  val: {val_dir}")
        sys.exit(1)

    # ============== 数据增强 ==============
    print("📦 初始化数据增强...")
    
    if args.use_augmentation:
        # ViT 推荐的增强策略
        train_transform = T.Compose([
            T.Resize((args.img_size + 32, args.img_size + 32), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(args.img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        print("  ✅ 训练集增强: Crop + Flip + ColorJitter + Blur + Erasing")
    else:
        train_transform = T.Compose([
            T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        print("  ✅ 训练集增强: 无")
    
    val_transform = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    # ============== 数据集 ==============
    print("\n📂 加载数据集...")
    try:
        train_ds = ViTDataset(str(train_dir), train_transform, use_grayscale=args.use_grayscale)
        print(f"  ✅ 训练集: {len(train_ds)} 张")
    except Exception as e:
        print(f"  ❌ 训练集加载失败: {e}")
        sys.exit(1)
    
    try:
        val_ds = ViTDataset(str(val_dir), val_transform, use_grayscale=args.use_grayscale)
        val_ds.class_to_idx = train_ds.class_to_idx
        val_ds.classes = train_ds.classes
        print(f"  ✅ 验证集: {len(val_ds)} 张")
    except Exception as e:
        print(f"  ❌ 验证集加载失败: {e}")
        sys.exit(1)
    
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"📊 类别: {class_names}")

    # ============== 数据分布统计 ==============
    print("\n📊 数据分布:")
    train_targets = [t for _, t in train_ds.samples]
    counts = Counter(train_targets)
    
    for cls_name in class_names:
        cls_idx = train_ds.class_to_idx[cls_name]
        count = counts[cls_idx]
        percentage = count / len(train_ds) * 100
        print(f"  {cls_name:10s}: {count:4d} 张 ({percentage:5.2f}%)")

    # ============== 类别平衡采样 ==============
    print("\n⚖️ 计算类别权重...")
    
    num_per_class = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    class_weight = num_per_class.sum() / torch.clamp(num_per_class, min=1.0)
    
    # 手动提升 JLC 权重（如果需要）
    jlc_idx = train_ds.class_to_idx.get('jlc')
    if jlc_idx is not None and counts[jlc_idx] < 500:
        class_weight[jlc_idx] *= 1.5
        print(f"  ⚠️ JLC 样本较少，权重已提升至: {class_weight[jlc_idx]:.2f}")
    
    sample_weight = torch.tensor([class_weight[t] for t in train_targets], dtype=torch.float)
    
    print(f"  类别权重: {[f'{w:.2f}' for w in class_weight.tolist()]}")
    
    sampler = WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(train_ds),
        replacement=True
    )

    # ============== DataLoader ==============
    print("\n🔄 创建 DataLoader...")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=False,
    )

    # ============== 模型 ==============
    print("\n🧠 加载模型...")
    
    # 选择 ViT 模型
    if args.model_type == "vit_b_16":
        model = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1)
        print("  📦 模型: ViT-B/16 (Base, patch_size=16)")
    elif args.model_type == "vit_b_32":
        model = tv.models.vit_b_32(weights=tv.models.ViT_B_32_Weights.IMAGENET1K_V1)
        print("  📦 模型: ViT-B/32 (Base, patch_size=32)")
    elif args.model_type == "vit_l_16":
        model = tv.models.vit_l_16(weights=tv.models.ViT_L_16_Weights.IMAGENET1K_V1)
        print("  📦 模型: ViT-L/16 (Large, patch_size=16)")
    else:
        print(f"  ❌ 未知模型类型: {args.model_type}")
        sys.exit(1)
    
    # 替换分类头
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ 可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  ✅ 总参数: {total_params / 1e6:.2f}M")

    # ============== 优化器 & 学习率 ==============
    print("\n⚙️ 配置优化器...")
    
    # AdamW 优化器（ViT 推荐）
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine Annealing 学习率调度
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup 阶段：线性增长
            return step / max(1, warmup_steps)
        else:
            # Cosine Annealing
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    cls_weight_tensor = class_weight.to(device)
    
    print(f"  ✅ 优化器: AdamW")
    print(f"  ✅ 初始学习率: {args.lr:.2e}")
    print(f"  ✅ Weight Decay: {args.weight_decay}")
    print(f"  ✅ Warmup Epochs: {args.warmup_epochs}")

    # ============== 训练循环 ==============
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    
    # 训练历史
    history = {
        'train_loss': [],        'val_loss': [],        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        # ===== 训练 =====
        model.train()
        t0 = time.time()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Mixup 增强（可选）
            if args.use_augmentation and hasattr(args, 'mixup_alpha') and args.mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, targets, args.mixup_alpha)
                
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    logits = model(images)
                    loss_func = lambda pred, y: focal_loss(pred, y, gamma=args.gamma, weight=cls_weight_tensor)
                    loss = mixup_criterion(loss_func, logits, targets_a, targets_b, lam)
            else:
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    logits = model(images)
                    loss = focal_loss(logits, targets, gamma=args.gamma, weight=cls_weight_tensor)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\r📝 Epoch {epoch}/{args.epochs} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss: {loss.item():.4f} | lr: {current_lr:.2e}", end='', flush=True)
        
        print()
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== 验证 =====
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                logits = model(images)
                loss = focal_loss(logits, targets, gamma=args.gamma, weight=cls_weight_tensor)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        
        acc = correct / total if total > 0 else 0.0
        f1 = macro_f1(y_true, y_pred, num_classes)
        avg_val_loss = val_loss / len(val_loader)
        
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['lr'].append(current_lr)
        
        print(f"✅ Epoch {epoch:03d}/{args.epochs} | "
              f"time: {elapsed:.1f}s | "
              f"train_loss: {avg_train_loss:.4f} | "
              f"val_loss: {avg_val_loss:.4f} | "
              f"lr: {current_lr:.2e} | "
              f"val_acc: {acc:.4f} | "
              f"macroF1: {f1:.4f} | "
              f"best: {best_acc:.4f}")
        
        # ===== 保存最佳模型 =====
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            no_improve = 0
            
            # 混淆矩阵
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
            
            # 保存模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "classes": class_names,
                "img_size": args.img_size,
                "mean": MEAN,
                "std": STD,
                "val_acc": acc,
                "macro_f1": f1,
                "model_type": args.model_type,
            }, out_dir / "classifier_best.pt")
            
            print(f"  💾 保存最佳模型 (acc={acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"⏹️  早停: {args.patience} 轮无提升")
                break
        
        print()
    
    # ===== 保存训练历史 =====
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线（分别保存为独立图片）
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 1. Train Loss vs Val Loss
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Validation Accuracy
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs_range, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'val_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Learning Rate Schedule
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(epochs_range, history['lr'], 'orange', label='Learning Rate', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig(out_dir / 'learning_rate.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # ===== 训练结束 =====
    print(f"\n{'='*70}")
    print(f"✅ 训练完成!")
    print(f"📊 最佳准确率: {best_acc:.4f} @ epoch {best_epoch}")
    print(f"💾 模型: {out_dir / 'classifier_best.pt'}")
    print(f"📈 损失曲线: {out_dir / 'loss_curves.png'}")
    print(f"📈 准确率曲线: {out_dir / 'val_accuracy.png'}")
    print(f"📈 学习率曲线: {out_dir / 'learning_rate.png'}")
    print(f"📊 混淆矩阵: {out_dir / 'confusion_matrix.png'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断训练")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)