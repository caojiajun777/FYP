"""
Task 1.1 - 生成性能指标与混淆矩阵
在测试集上评估训练好的 ViT 模型
"""

import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from collections import Counter

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# 配置
MODEL_PATH = os.environ.get("TASK1_VIT_MODEL_PATH", r"D:\FYP\runs_vit\train_vit_b16_best\classifier_best.pt")
DATA_ROOT = os.environ.get("TASK1_DATA_ROOT", r"D:\FYP\data\EDA_cls_dataset")
OUTPUT_DIR = os.environ.get("TASK1_OUTPUT_DIR", r"D:\FYP\Classifier\paper_results\task1_source_classification")

CLASSES = ["altium", "kicad", "orcad", "eagle", "jlc"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的 ViT 模型"""
    print(f"\n📦 加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    model_type = checkpoint.get('model_type', 'vit_b_16')
    num_classes = len(checkpoint['classes'])
    img_size = checkpoint.get('img_size', 224)
    
    print(f"  模型类型: {model_type}")
    print(f"  类别数: {num_classes}")
    print(f"  图像尺寸: {img_size}")
    
    # 创建模型
    if model_type == "vit_b_16":
        model = tv.models.vit_b_16(weights=None)
    elif model_type == "vit_b_32":
        model = tv.models.vit_b_32(weights=None)
    elif model_type == "vit_l_16":
        model = tv.models.vit_l_16(weights=None)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    # 替换分类头
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"  训练 Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  验证准确率: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"  Macro F1: {checkpoint.get('macro_f1', 'N/A'):.4f}")
    
    return model, checkpoint


def evaluate_model(model, dataloader, device='cuda'):
    """评估模型"""
    print(f"\n🔍 评估模型...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    all_logits = np.vstack(all_logits)
    
    print(f"✅ 评估完成: {len(all_labels)} 个样本")
    
    return np.array(all_preds), np.array(all_labels), all_logits


def compute_metrics(y_true, y_pred, class_names):
    """计算性能指标"""
    print(f"\n📊 计算性能指标...")
    
    # 整体指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # 每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 宏平均
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # 加权平均
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
        },
        'per_class': {}
    }
    
    # 每个类别的详细指标
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # 打印
    print(f"\n{'='*60}")
    print(f"整体性能:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  宏平均 F1 (Macro F1): {macro_f1:.4f}")
    print(f"  加权平均 F1 (Weighted F1): {weighted_f1:.4f}")
    print(f"{'='*60}")
    
    print(f"\n每类别性能:")
    print(f"{'类别':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print('-'*60)
    for class_name in class_names:
        m = metrics['per_class'][class_name]
        print(f"{class_name:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10d}")
    print('-'*60)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, output_path, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    print(f"\n📈 绘制混淆矩阵...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绝对数量
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, square=True)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Counts)', fontsize=14, fontweight='bold')
    
    # 百分比
    ax = axes[1]
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, square=True)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存: {output_path}")
    
    return cm


def save_metrics_json(metrics, output_path):
    """保存指标为 JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 指标已保存: {output_path}")


def generate_report(metrics, cm, class_names, output_path):
    """生成 Markdown 报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Task 1.1 - ViT 模型性能评估报告\n\n")
        
        # 整体性能
        f.write("## 1. 整体性能\n\n")
        overall = metrics['overall']
        f.write("| 指标 | 数值 |\n")
        f.write("|------|------|\n")
        f.write(f"| 准确率 (Accuracy) | {overall['accuracy']:.4f} |\n")
        f.write(f"| 宏平均精确率 (Macro Precision) | {overall['macro_precision']:.4f} |\n")
        f.write(f"| 宏平均召回率 (Macro Recall) | {overall['macro_recall']:.4f} |\n")
        f.write(f"| 宏平均 F1 (Macro F1) | {overall['macro_f1']:.4f} |\n")
        f.write(f"| 加权精确率 (Weighted Precision) | {overall['weighted_precision']:.4f} |\n")
        f.write(f"| 加权召回率 (Weighted Recall) | {overall['weighted_recall']:.4f} |\n")
        f.write(f"| 加权 F1 (Weighted F1) | {overall['weighted_f1']:.4f} |\n\n")
        
        # 每类别性能
        f.write("## 2. 每类别性能\n\n")
        f.write("| 类别 | Precision | Recall | F1-Score | Support |\n")
        f.write("|------|-----------|--------|----------|----------|\n")
        for class_name in class_names:
            m = metrics['per_class'][class_name]
            f.write(f"| {class_name} | {m['precision']:.4f} | {m['recall']:.4f} | "
                   f"{m['f1']:.4f} | {m['support']} |\n")
        f.write("\n")
        
        # 混淆矩阵分析
        f.write("## 3. 混淆矩阵分析\n\n")
        f.write("### 3.1 混淆矩阵（数量）\n\n")
        f.write("| 真实\\预测 |")
        for cn in class_names:
            f.write(f" {cn} |")
        f.write("\n|")
        f.write("---|" * (len(class_names) + 1))
        f.write("\n")
        
        for i, true_class in enumerate(class_names):
            f.write(f"| **{true_class}** |")
            for j in range(len(class_names)):
                f.write(f" {cm[i, j]} |")
            f.write("\n")
        f.write("\n")
        
        # 常见混淆对
        f.write("### 3.2 主要混淆分析\n\n")
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((class_names[i], class_names[j], cm[i, j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if confusion_pairs:
            f.write("**最常见的混淆情况**（真实 → 预测）：\n\n")
            for true_cls, pred_cls, count in confusion_pairs[:5]:
                f.write(f"- {true_cls} → {pred_cls}: {count} 次\n")
        else:
            f.write("无混淆（完美分类）\n")
        
        f.write("\n---\n")
        f.write(f"生成时间: 2025-01-19\n")
    
    print(f"✅ 评估报告已保存: {output_path}")


def main():
    print("\n" + "="*60)
    print("Task 1.1 - ViT 模型性能评估")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 设备: {device}")
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    metrics_dir = output_dir / 'metrics'
    viz_dir = output_dir / 'visualizations'
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model, checkpoint = load_model(MODEL_PATH, device)
    img_size = checkpoint.get('img_size', 224)
    
    # 准备数据
    print(f"\n📂 加载测试集...")
    val_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    
    val_dir = Path(DATA_ROOT) / 'test'
    val_dataset = ImageFolder(str(val_dir), transform=val_transform)
    
    # 确保类别顺序一致
    val_dataset.class_to_idx = {cls: i for i, cls in enumerate(checkpoint['classes'])}
    val_dataset.classes = checkpoint['classes']
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    print(f"✅ 测试集: {len(val_dataset)} 张")
    print(f"  类别: {val_dataset.classes}")
    
    # 评估
    y_pred, y_true, logits = evaluate_model(model, val_loader, device)
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred, CLASSES)
    
    # 绘制混淆矩阵
    cm = plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        viz_dir / 'confusion_matrix_test.png',
        title='Test Set Confusion Matrix'
    )
    
    # 保存结果
    print(f"\n💾 保存结果...")
    save_metrics_json(metrics, metrics_dir / 'test_metrics.json')
    generate_report(metrics, cm, CLASSES, metrics_dir / 'test_evaluation_report.md')
    
    # 保存预测结果
    predictions = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'class_names': CLASSES
    }
    with open(metrics_dir / 'test_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✅ 预测结果已保存: {metrics_dir / 'test_predictions.json'}")
    
    print(f"\n{'='*60}")
    print("✅ Task 1.1 完成!")
    print('='*60)
    print(f"📁 输出目录:")
    print(f"  - {metrics_dir / 'test_metrics.json'}")
    print(f"  - {metrics_dir / 'test_evaluation_report.md'}")
    print(f"  - {metrics_dir / 'test_predictions.json'}")
    print(f"  - {viz_dir / 'confusion_matrix_test.png'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
