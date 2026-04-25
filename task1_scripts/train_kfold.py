"""
Stratified 3-Fold Cross-Validation for EDA Schematic Classification

Requirements:
- Reproducible CV pipeline with consistent label mappings
- Fold-level and aggregated metrics/figures for paper
- Early stopping on validation split (not test fold)
- Extensible to multiple models (ResNet50, ViT)

Usage:
    python train_kfold.py --data_root <path> --model resnet50 --folds 3 --seed 42
"""

import os
import sys
import json
import csv
import math
import time
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration & Defaults
# ============================================================

CLASSES = ["altium", "eagle", "jlc", "kicad", "orcad"]  # Alphabetical order
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.406]

DEFAULT_CONFIG = {
    'data_root': os.environ.get('TASK1_KFOLD_DATA_ROOT', r'D:\FYP\data\EDA_cls_dataset_kfold'),
    'model': 'resnet50',
    'folds': 3,
    'seed': 42,
    'epochs': 30,
    'batch_size': 32,
    'lr': 1.5e-4,
    'weight_decay': 0.05,
    'patience': 8,
    'val_split': 0.1,  # 10% of train for validation
    'img_size': 224,
    'num_workers': 4,
    'gamma': 2.0,  # Focal loss gamma
    'mixup_alpha': 0.2,
    'use_weighted_sampler': True,
    'out_dir': os.environ.get('TASK1_KFOLD_MODEL_ROOT', r'D:\FYP\runs_kfold')
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"🔧 Random seed set to: {seed}")


# ============================================================
# Loss Functions
# ============================================================

def focal_loss(logits, targets, gamma=2.0, weight=None):
    """Focal Loss for handling class imbalance"""
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=weight)
    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# Model Creation
# ============================================================

def create_model(model_name, num_classes=5):
    """Create model with pretrained weights"""
    if model_name == 'resnet50':
        model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "ResNet50"
    
    elif model_name == 'vit_b_16':
        model = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, "ViT-B/16"
    
    elif model_name == 'convnext_tiny':
        model = tv.models.convnext_tiny(weights=tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model, "ConvNeXt-Tiny"
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Data Transforms
# ============================================================

def get_transforms(img_size, augment=True):
    """Get train and test transforms"""
    if augment:
        train_transform = T.Compose([
            T.Resize((img_size + 32, img_size + 32), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.3),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    else:
        train_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])
    
    test_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])
    
    return train_transform, test_transform


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_model(model, loader, device, num_classes, idx_to_class):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                logits = model(images)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Overall metrics
    acc = accuracy_score(all_targets, all_preds)
    
    # Use explicit label order for consistency
    labels = list(range(num_classes))
    target_names = [idx_to_class[i] for i in labels]
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, average=None, zero_division=0
    )
    
    # Macro averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    
    # Classification report
    report = classification_report(
        all_targets, all_preds, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0
    )
    
    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'per_class': {
            target_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            } for i in range(num_classes)
        },
        'confusion_matrix': cm.tolist(),
        'y_true': all_targets.tolist(),
        'y_pred': all_preds.tolist(),
        'y_probs': all_probs.tolist()
    }
    
    return metrics, cm


# ============================================================
# Plotting Functions
# ============================================================

def plot_confusion_matrix(cm, class_names, save_path, normalize=False, title='Confusion Matrix'):
    """Plot confusion matrix"""
    if normalize:
        cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        cmap = 'Blues'
    else:
        cm_plot = cm
        fmt = 'd'
        cmap = 'Blues'
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized' if normalize else 'Count'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_misclassified(metrics, dataset, fold_dir, idx_to_class):
    """Save misclassified samples to CSV"""
    y_true = np.array(metrics['y_true'])
    y_pred = np.array(metrics['y_pred'])
    y_probs = np.array(metrics['y_probs'])
    
    misclassified = []
    for i, (gt, pred) in enumerate(zip(y_true, y_pred)):
        if gt != pred:
            # Get confidence of predicted class
            confidence = y_probs[i][pred]
            misclassified.append({
                'index': i,
                'gt_label': idx_to_class[gt],
                'pred_label': idx_to_class[pred],
                'confidence': f'{confidence:.4f}'
            })
    
    csv_path = fold_dir / 'misclassified.csv'
    with open(csv_path, 'w', newline='') as f:
        if misclassified:
            writer = csv.DictWriter(f, fieldnames=['index', 'gt_label', 'pred_label', 'confidence'])
            writer.writeheader()
            writer.writerows(misclassified)
    
    print(f"  📄 Misclassified samples: {len(misclassified)} (saved to {csv_path.name})")


# ============================================================
# Training Loop
# ============================================================

def train_one_fold(fold_id, train_idx, test_idx, full_dataset, config, device):
    """Train and evaluate one fold"""
    
    print(f"\n{'='*70}")
    print(f"📊 FOLD {fold_id + 1}/{config['folds']}")
    print('='*70)
    
    # Create fold directory
    fold_dir = Path(config['out_dir']) / config['model'] / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset mappings
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # Split train into train+val (stratified)
    train_targets = [full_dataset.targets[i] for i in train_idx]
    train_train_idx, train_val_idx = train_test_split(
        train_idx, test_size=config['val_split'], stratify=train_targets,
        random_state=config['seed']
    )
    
    print(f"📂 Data splits:")
    print(f"  Train: {len(train_train_idx)} samples")
    print(f"  Val:   {len(train_val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    # Count per-class samples
    def count_classes(indices):
        targets = [full_dataset.targets[i] for i in indices]
        return Counter(targets)
    
    train_counts = count_classes(train_train_idx)
    val_counts = count_classes(train_val_idx)
    test_counts = count_classes(test_idx)
    
    print(f"\n📊 Class distribution:")
    print(f"{'Class':<10} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 40)
    for cls_name in CLASSES:
        cls_idx = class_to_idx[cls_name]
        print(f"{cls_name:<10} {train_counts[cls_idx]:<8} {val_counts[cls_idx]:<8} {test_counts[cls_idx]:<8}")
    
    # Create data subsets
    train_transform, test_transform = get_transforms(config['img_size'], augment=True)
    
    # Clone dataset with different transforms
    train_dataset = ImageFolder(config['data_root'], transform=train_transform)
    val_dataset = ImageFolder(config['data_root'], transform=test_transform)
    test_dataset = ImageFolder(config['data_root'], transform=test_transform)
    
    train_subset = Subset(train_dataset, train_train_idx)
    val_subset = Subset(val_dataset, train_val_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    # Create weighted sampler for training
    if config['use_weighted_sampler']:
        class_counts_tensor = torch.tensor([train_counts[i] for i in range(num_classes)], dtype=torch.float)
        class_weights = class_counts_tensor.sum() / torch.clamp(class_counts_tensor, min=1.0)
        sample_weights = torch.tensor([class_weights[full_dataset.targets[i]] for i in train_train_idx])
        sampler = WeightedRandomSampler(sample_weights, len(train_train_idx), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], 
                             sampler=sampler, shuffle=shuffle,
                             num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)
    
    # Create model
    model, model_name = create_model(config['model'], num_classes)
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
    
    total_steps = config['epochs'] * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    # Class weights for loss
    class_weights_tensor = class_weights.to(device) if config['use_weighted_sampler'] else None
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    best_ckpt_path = fold_dir / 'best_model.pt'
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Mixup
            if config['mixup_alpha'] > 0:
                images, targets_a, targets_b, lam = mixup_data(images, targets, config['mixup_alpha'])
                
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    logits = model(images)
                    loss_fn = lambda pred, y: focal_loss(pred, y, config['gamma'], class_weights_tensor)
                    loss = mixup_criterion(loss_fn, logits, targets_a, targets_b, lam)
            else:
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    logits = model(images)
                    loss = focal_loss(logits, targets, config['gamma'], class_weights_tensor)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        val_metrics, _ = evaluate_model(model, val_loader, device, num_classes, idx_to_class)
        
        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'class_to_idx': class_to_idx
            }, best_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= config['patience']:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break
    
    print(f"✅ Best validation F1: {best_val_f1:.4f} @ epoch {best_epoch}")
    
    # Load best model and evaluate on test set
    print(f"\n🔍 Evaluating on test fold...")
    checkpoint = torch.load(best_ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_cm = evaluate_model(model, test_loader, device, num_classes, idx_to_class)
    
    print(f"\n📊 Test Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Macro-P:   {test_metrics['macro_precision']:.4f}")
    print(f"  Macro-R:   {test_metrics['macro_recall']:.4f}")
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # 1. Metrics JSON
    fold_summary = {
        'fold_id': fold_id,
        'model': config['model'],
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'checkpoint_path': str(best_ckpt_path),
        'dataset_splits': {
            'train': len(train_train_idx),
            'val': len(train_val_idx),
            'test': len(test_idx)
        },
        'class_distribution': {
            'train': {idx_to_class[k]: v for k, v in train_counts.items()},
            'val': {idx_to_class[k]: v for k, v in val_counts.items()},
            'test': {idx_to_class[k]: v for k, v in test_counts.items()}
        },
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'test_metrics': test_metrics
    }
    
    with open(fold_dir / 'metrics.json', 'w') as f:
        json.dump(fold_summary, f, indent=2)
    
    # 2. Confusion matrices
    np.save(fold_dir / 'confusion_matrix_raw.npy', test_cm)
    
    plot_confusion_matrix(test_cm, CLASSES, fold_dir / 'confusion_matrix_raw.png',
                         normalize=False, title=f'Fold {fold_id+1} - Confusion Matrix (Counts)')
    
    plot_confusion_matrix(test_cm, CLASSES, fold_dir / 'confusion_matrix_norm.png',
                         normalize=True, title=f'Fold {fold_id+1} - Confusion Matrix (Normalized)')
    
    # 3. Misclassified samples
    save_misclassified(test_metrics, test_subset, fold_dir, idx_to_class)
    
    print(f"  ✅ Saved to: {fold_dir}")
    
    return fold_summary, test_cm


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stratified K-Fold Cross-Validation')
    parser.add_argument('--data_root', type=str, default=DEFAULT_CONFIG['data_root'])
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'],
                       choices=['resnet50', 'vit_b_16', 'convnext_tiny'])
    parser.add_argument('--folds', type=int, default=DEFAULT_CONFIG['folds'])
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--out_dir', type=str, default=DEFAULT_CONFIG['out_dir'])
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))
    
    # Set seed
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print(f"🚀 {config['model'].upper()} - Stratified {config['folds']}-Fold Cross-Validation")
    print("="*70)
    print(f"📁 Data: {config['data_root']}")
    print(f"🔧 Device: {device}")
    print(f"🔧 Seed: {config['seed']}")
    print(f"🔧 Folds: {config['folds']}")
    print(f"📁 Output: {config['out_dir']}/{config['model']}/")
    print("="*70)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    full_dataset = ImageFolder(config['data_root'])
    
    print(f"  ✅ Total samples: {len(full_dataset)}")
    print(f"  ✅ Classes: {full_dataset.classes}")
    print(f"  ✅ class_to_idx: {full_dataset.class_to_idx}")
    
    # Verify class counts
    class_counts = Counter(full_dataset.targets)
    for cls_name in CLASSES:
        cls_idx = full_dataset.class_to_idx[cls_name]
        print(f"    {cls_name}: {class_counts[cls_idx]}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=config['folds'], shuffle=True, random_state=config['seed'])
    
    all_fold_results = []
    all_cms = []
    
    # Train each fold
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(full_dataset.targets)), 
                                                               full_dataset.targets)):
        fold_summary, fold_cm = train_one_fold(fold_id, train_idx, test_idx, 
                                               full_dataset, config, device)
        all_fold_results.append(fold_summary)
        all_cms.append(fold_cm)
    
    # ============================================================
    # Aggregate Results
    # ============================================================
    
    print(f"\n{'='*70}")
    print("📊 AGGREGATING RESULTS")
    print('='*70)
    
    out_dir = Path(config['out_dir']) / config['model']
    
    # Extract metrics
    accuracies = [r['test_metrics']['accuracy'] for r in all_fold_results]
    macro_f1s = [r['test_metrics']['macro_f1'] for r in all_fold_results]
    macro_ps = [r['test_metrics']['macro_precision'] for r in all_fold_results]
    macro_rs = [r['test_metrics']['macro_recall'] for r in all_fold_results]
    
    # Per-class metrics
    per_class_recalls = defaultdict(list)
    per_class_f1s = defaultdict(list)
    per_class_precisions = defaultdict(list)
    
    for result in all_fold_results:
        for cls_name, metrics in result['test_metrics']['per_class'].items():
            per_class_recalls[cls_name].append(metrics['recall'])
            per_class_f1s[cls_name].append(metrics['f1'])
            per_class_precisions[cls_name].append(metrics['precision'])
    
    # Summary statistics
    cv_summary = {
        'model': config['model'],
        'n_folds': config['folds'],
        'seed': config['seed'],
        'overall_metrics': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'macro_f1_mean': float(np.mean(macro_f1s)),
            'macro_f1_std': float(np.std(macro_f1s)),
            'macro_precision_mean': float(np.mean(macro_ps)),
            'macro_precision_std': float(np.std(macro_ps)),
            'macro_recall_mean': float(np.mean(macro_rs)),
            'macro_recall_std': float(np.std(macro_rs))
        },
        'per_class_metrics': {}
    }
    
    for cls_name in CLASSES:
        cv_summary['per_class_metrics'][cls_name] = {
            'recall_mean': float(np.mean(per_class_recalls[cls_name])),
            'recall_std': float(np.std(per_class_recalls[cls_name])),
            'f1_mean': float(np.mean(per_class_f1s[cls_name])),
            'f1_std': float(np.std(per_class_f1s[cls_name])),
            'precision_mean': float(np.mean(per_class_precisions[cls_name])),
            'precision_std': float(np.std(per_class_precisions[cls_name]))
        }
    
    # Save summary JSON
    with open(out_dir / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    # Save summary CSV
    with open(out_dir / 'cv_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std'])
        writer.writerow(['Accuracy', f"{cv_summary['overall_metrics']['accuracy_mean']:.4f}", 
                        f"{cv_summary['overall_metrics']['accuracy_std']:.4f}"])
        writer.writerow(['Macro-F1', f"{cv_summary['overall_metrics']['macro_f1_mean']:.4f}",
                        f"{cv_summary['overall_metrics']['macro_f1_std']:.4f}"])
        writer.writerow(['Macro-Precision', f"{cv_summary['overall_metrics']['macro_precision_mean']:.4f}",
                        f"{cv_summary['overall_metrics']['macro_precision_std']:.4f}"])
        writer.writerow(['Macro-Recall', f"{cv_summary['overall_metrics']['macro_recall_mean']:.4f}",
                        f"{cv_summary['overall_metrics']['macro_recall_std']:.4f}"])
    
    # Per-class summary CSV
    with open(out_dir / 'per_class_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision (mean±std)', 'Recall (mean±std)', 'F1 (mean±std)'])
        for cls_name in CLASSES:
            m = cv_summary['per_class_metrics'][cls_name]
            writer.writerow([
                cls_name,
                f"{m['precision_mean']:.4f}±{m['precision_std']:.4f}",
                f"{m['recall_mean']:.4f}±{m['recall_std']:.4f}",
                f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}"
            ])
    
    # Aggregated confusion matrices
    cm_aggregated = np.sum(all_cms, axis=0)
    np.save(out_dir / 'cm_aggregated.npy', cm_aggregated)
    
    plot_confusion_matrix(cm_aggregated, CLASSES, out_dir / 'cm_aggregated.png',
                         normalize=False, title='Aggregated Confusion Matrix (All Folds)')
    
    plot_confusion_matrix(cm_aggregated, CLASSES, out_dir / 'cm_aggregated_normalized.png',
                         normalize=True, title='Aggregated Confusion Matrix (Normalized)')
    
    # Print summary
    print(f"\n📊 Overall Performance:")
    print(f"  Accuracy:  {cv_summary['overall_metrics']['accuracy_mean']:.4f} ± {cv_summary['overall_metrics']['accuracy_std']:.4f}")
    print(f"  Macro-F1:  {cv_summary['overall_metrics']['macro_f1_mean']:.4f} ± {cv_summary['overall_metrics']['macro_f1_std']:.4f}")
    print(f"  Macro-P:   {cv_summary['overall_metrics']['macro_precision_mean']:.4f} ± {cv_summary['overall_metrics']['macro_precision_std']:.4f}")
    print(f"  Macro-R:   {cv_summary['overall_metrics']['macro_recall_mean']:.4f} ± {cv_summary['overall_metrics']['macro_recall_std']:.4f}")
    
    print(f"\n📊 Per-Class Performance:")
    print(f"{'Class':<12} {'Precision':<20} {'Recall':<20} {'F1-Score':<20}")
    print("-" * 72)
    for cls_name in CLASSES:
        m = cv_summary['per_class_metrics'][cls_name]
        print(f"{cls_name:<12} "
              f"{m['precision_mean']:.4f}±{m['precision_std']:.4f}      "
              f"{m['recall_mean']:.4f}±{m['recall_std']:.4f}      "
              f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✅ {config['folds']}-Fold Cross-Validation Complete!")
    print(f"📁 Results saved to: {out_dir}/")
    print(f"  - cv_summary.json")
    print(f"  - cv_summary.csv")
    print(f"  - per_class_summary.csv")
    print(f"  - cm_aggregated.png")
    print(f"  - cm_aggregated_normalized.png")
    print(f"  - fold0/, fold1/, fold2/ (individual fold results)")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
