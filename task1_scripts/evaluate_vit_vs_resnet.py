"""
在测试集上评估 ViT vs ResNet 的多标签分类性能
"""

import os
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, hamming_loss, 
    multilabel_confusion_matrix, accuracy_score,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as T

# ============ 配置 ============
DATASET_CSV = Path('task2_function_prediction_pure_vl/core_dataset_pure_vl.csv')
SOURCE_IMAGE_DIR = Path(os.environ.get("TASK1_DATA_ROOT", r'D:\FYP\data\EDA_cls_dataset'))
MODEL_DIR = Path('task2_core_classification_multilabel')
OUTPUT_DIR = Path('task2_comparison_results')
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_CLASSES = 3

CLASS_NAMES = ['Evaluation_Board', 'Power_Management', 'Interface']

# 模型路径
VIT_MODEL_PATH = MODEL_DIR / 'models_vit' / 'best_model.pt'
RESNET_MODEL_PATH = MODEL_DIR / 'models_resnet' / 'best_model_resnet.pt'


class MultiLabelCircuitDataset(Dataset):
    """多标签电路功能数据集"""
    
    def __init__(self, df, source_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.source_dir = source_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 读取图片
        img_path = self.source_dir / row['split'] / row['eda_tool'] / row['filename']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # 多标签（3个类别的二进制向量）
        labels = torch.tensor([
            row['core_eval_board'],
            row['core_power_management'],
            row['core_interface']
        ], dtype=torch.float32)
        
        return image, labels, row['filename']


def create_test_dataloader(img_size=224):
    """创建测试数据加载器"""
    print("=" * 80)
    print("加载测试数据集")
    print("=" * 80)
    
    df = pd.read_csv(DATASET_CSV)
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"\n测试集大小: {len(test_df)} 样本")
    print(f"\n测试集标签分布:")
    print(f"  - Evaluation Board: {test_df['core_eval_board'].sum()}")
    print(f"  - Power Management: {test_df['core_power_management'].sum()}")
    print(f"  - Interface: {test_df['core_interface'].sum()}")
    
    label_sum = test_df[['core_eval_board', 'core_power_management', 'core_interface']].sum(axis=1)
    print(f"\n多标签分布:")
    print(f"  - 单标签: {int((label_sum == 1).sum())} ({(label_sum == 1).sum()/len(test_df)*100:.1f}%)")
    print(f"  - 双标签: {int((label_sum == 2).sum())} ({(label_sum == 2).sum()/len(test_df)*100:.1f}%)")
    print(f"  - 三标签: {int((label_sum == 3).sum())} ({(label_sum == 3).sum()/len(test_df)*100:.1f}%)")
    
    # 标准化变换
    test_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = MultiLabelCircuitDataset(test_df, SOURCE_IMAGE_DIR, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, test_df


def load_vit_model(checkpoint_path):
    """加载 ViT 模型"""
    print("\n" + "=" * 80)
    print("加载 ViT-B/16 模型")
    print("=" * 80)
    
    # 创建模型
    model = tv.models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ 模型加载成功")
    val_f1 = checkpoint.get('val_macro_f1', None)
    if val_f1 is not None:
        print(f"  - Val Macro-F1: {val_f1:.4f}")
    else:
        print(f"  - Val Macro-F1: N/A")
    
    return model, checkpoint


def load_resnet_model(checkpoint_path):
    """加载 ResNet50 模型"""
    print("\n" + "=" * 80)
    print("加载 ResNet50 模型")
    print("=" * 80)
    
    # 创建模型
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ 模型加载成功")
    val_f1 = checkpoint.get('val_macro_f1', None)
    if val_f1 is not None:
        print(f"  - Val Macro-F1: {val_f1:.4f}")
    else:
        print(f"  - Val Macro-F1: N/A")
    
    return model, checkpoint


def evaluate_model(model, test_loader, model_name):
    """评估模型"""
    print("\n" + "=" * 80)
    print(f"评估 {model_name} 在测试集上的性能")
    print("=" * 80)
    
    all_preds = []
    all_labels = []
    all_filenames = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(DEVICE)
            outputs = model(images)
            
            # 预测概率和二值化预测
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_filenames.extend(filenames)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # 计算指标
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    
    # 样本级别准确率（exact match）
    exact_match = accuracy_score(all_labels, all_preds)
    
    # Per-class 指标
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrices
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    
    # 打印结果
    print(f"\n{model_name} 测试结果:")
    print(f"  Exact Match (样本级准确率): {exact_match:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    
    print(f"\nPer-class 指标:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"\n{class_name}:")
        print(f"  F1:        {per_class_f1[i]:.4f}")
        print(f"  Precision: {per_class_precision[i]:.4f}")
        print(f"  Recall:    {per_class_recall[i]:.4f}")
    
    results = {
        'model_name': model_name,
        'exact_match': float(exact_match),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'hamming_loss': float(hamming),
        'per_class_metrics': {
            CLASS_NAMES[i]: {
                'f1': float(per_class_f1[i]),
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i])
            } for i in range(NUM_CLASSES)
        },
        'confusion_matrices': [cm.tolist() for cm in conf_matrices]
    }
    
    return results, all_preds, all_labels, all_probs, all_filenames


def plot_comparison(vit_results, resnet_results):
    """绘制对比图表"""
    print("\n" + "=" * 80)
    print("生成对比可视化")
    print("=" * 80)
    
    # 1. 总体指标对比
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    metrics = ['exact_match', 'macro_f1', 'micro_f1', 'hamming_loss']
    metric_names = ['Exact Match', 'Macro-F1', 'Micro-F1', 'Hamming Loss']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        vit_val = vit_results[metric]
        resnet_val = resnet_results[metric]
        
        # Hamming Loss 越小越好，其他越大越好
        if metric == 'hamming_loss':
            better = 'ViT' if vit_val < resnet_val else 'ResNet'
            colors = ['green' if vit_val < resnet_val else 'orange', 
                     'orange' if vit_val < resnet_val else 'green']
        else:
            better = 'ViT' if vit_val > resnet_val else 'ResNet'
            colors = ['green' if vit_val > resnet_val else 'orange',
                     'orange' if vit_val > resnet_val else 'green']
        
        bars = axes[idx].bar(['ViT', 'ResNet'], [vit_val, resnet_val], color=colors)
        axes[idx].set_ylabel(name)
        axes[idx].set_title(f'{name}\n(Better: {better})')
        axes[idx].set_ylim([0, max(vit_val, resnet_val) * 1.2])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: overall_comparison.png")
    
    # 2. Per-class F1 对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    vit_f1s = [vit_results['per_class_metrics'][cls]['f1'] for cls in CLASS_NAMES]
    resnet_f1s = [resnet_results['per_class_metrics'][cls]['f1'] for cls in CLASS_NAMES]
    
    bars1 = ax.bar(x - width/2, vit_f1s, width, label='ViT', color='steelblue')
    bars2 = ax.bar(x + width/2, resnet_f1s, width, label='ResNet', color='coral')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: per_class_f1_comparison.png")
    
    # 3. Per-class Precision/Recall 对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Precision
    vit_prec = [vit_results['per_class_metrics'][cls]['precision'] for cls in CLASS_NAMES]
    resnet_prec = [resnet_results['per_class_metrics'][cls]['precision'] for cls in CLASS_NAMES]
    
    bars1 = axes[0].bar(x - width/2, vit_prec, width, label='ViT', color='steelblue')
    bars2 = axes[0].bar(x + width/2, resnet_prec, width, label='ResNet', color='coral')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Per-Class Precision Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Recall
    vit_rec = [vit_results['per_class_metrics'][cls]['recall'] for cls in CLASS_NAMES]
    resnet_rec = [resnet_results['per_class_metrics'][cls]['recall'] for cls in CLASS_NAMES]
    
    bars1 = axes[1].bar(x - width/2, vit_rec, width, label='ViT', color='steelblue')
    bars2 = axes[1].bar(x + width/2, resnet_rec, width, label='ResNet', color='coral')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Per-Class Recall Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: per_class_precision_recall_comparison.png")


def generate_comparison_report(vit_results, resnet_results):
    """生成对比报告"""
    print("\n" + "=" * 80)
    print("生成详细对比报告")
    print("=" * 80)
    
    report = []
    report.append("# ViT vs ResNet 测试集性能对比报告\n")
    report.append(f"数据集: Task 2 Function Prediction (Multi-label)\n")
    report.append(f"评估时间: {pd.Timestamp.now()}\n")
    report.append("\n## 总体性能对比\n")
    report.append("| 指标 | ViT-B/16 | ResNet50 | 差异 | 更优 |\n")
    report.append("|------|----------|----------|------|------|\n")
    
    metrics = [
        ('exact_match', 'Exact Match (样本级准确率)', False),
        ('macro_f1', 'Macro-F1', False),
        ('micro_f1', 'Micro-F1', False),
        ('hamming_loss', 'Hamming Loss', True)  # True means lower is better
    ]
    
    for key, name, lower_better in metrics:
        vit_val = vit_results[key]
        resnet_val = resnet_results[key]
        diff = vit_val - resnet_val
        
        if lower_better:
            better = 'ViT' if vit_val < resnet_val else 'ResNet'
            diff_str = f"{-diff:.4f}" if diff < 0 else f"+{-diff:.4f}"
        else:
            better = 'ViT' if vit_val > resnet_val else 'ResNet'
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        
        report.append(f"| {name} | {vit_val:.4f} | {resnet_val:.4f} | {diff_str} | **{better}** |\n")
    
    report.append("\n## Per-Class 性能对比\n")
    
    for class_name in CLASS_NAMES:
        report.append(f"\n### {class_name}\n")
        report.append("| 指标 | ViT-B/16 | ResNet50 | 差异 | 更优 |\n")
        report.append("|------|----------|----------|------|------|\n")
        
        for metric_name in ['f1', 'precision', 'recall']:
            vit_val = vit_results['per_class_metrics'][class_name][metric_name]
            resnet_val = resnet_results['per_class_metrics'][class_name][metric_name]
            diff = vit_val - resnet_val
            better = 'ViT' if vit_val > resnet_val else 'ResNet'
            diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
            
            metric_display = metric_name.capitalize()
            report.append(f"| {metric_display} | {vit_val:.4f} | {resnet_val:.4f} | {diff_str} | **{better}** |\n")
    
    report.append("\n## 关键发现\n\n")
    
    # 判断哪个模型更好
    vit_wins = 0
    resnet_wins = 0
    
    for key, _, lower_better in metrics:
        vit_val = vit_results[key]
        resnet_val = resnet_results[key]
        if lower_better:
            if vit_val < resnet_val:
                vit_wins += 1
            else:
                resnet_wins += 1
        else:
            if vit_val > resnet_val:
                vit_wins += 1
            else:
                resnet_wins += 1
    
    if vit_wins > resnet_wins:
        report.append(f"- **ViT-B/16** 在测试集上表现更优，在 {vit_wins}/{vit_wins+resnet_wins} 项总体指标上领先\n")
    elif resnet_wins > vit_wins:
        report.append(f"- **ResNet50** 在测试集上表现更优，在 {resnet_wins}/{vit_wins+resnet_wins} 项总体指标上领先\n")
    else:
        report.append(f"- **两个模型表现相当**，各有优势\n")
    
    # 分析每个类别的优势
    report.append("\n### 各类别优势分析:\n\n")
    for class_name in CLASS_NAMES:
        vit_f1 = vit_results['per_class_metrics'][class_name]['f1']
        resnet_f1 = resnet_results['per_class_metrics'][class_name]['f1']
        
        if vit_f1 > resnet_f1:
            report.append(f"- **{class_name}**: ViT 更优 (F1: {vit_f1:.4f} vs {resnet_f1:.4f}, +{vit_f1-resnet_f1:.4f})\n")
        elif resnet_f1 > vit_f1:
            report.append(f"- **{class_name}**: ResNet 更优 (F1: {resnet_f1:.4f} vs {vit_f1:.4f}, +{resnet_f1-vit_f1:.4f})\n")
        else:
            report.append(f"- **{class_name}**: 两者相当 (F1: {vit_f1:.4f})\n")
    
    report.append("\n## 模型参数对比\n\n")
    report.append("| 模型 | 参数量 | 架构类型 | 预训练权重 |\n")
    report.append("|------|--------|----------|------------|\n")
    report.append("| ViT-B/16 | ~86M | Transformer | ImageNet-1K |\n")
    report.append("| ResNet50 | ~23.5M | CNN | ImageNet-1K |\n")
    
    report_text = ''.join(report)
    
    with open(OUTPUT_DIR / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 保存: comparison_report.md")
    
    return report_text


def main():
    print("=" * 80)
    print("ViT vs ResNet 测试集性能对比")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"ViT Model: {VIT_MODEL_PATH}")
    print(f"ResNet Model: {RESNET_MODEL_PATH}")
    
    # 1. 加载测试数据
    test_loader, test_df = create_test_dataloader()
    
    # 2. 加载模型
    vit_model, vit_checkpoint = load_vit_model(VIT_MODEL_PATH)
    resnet_model, resnet_checkpoint = load_resnet_model(RESNET_MODEL_PATH)
    
    # 3. 评估 ViT
    vit_results, vit_preds, vit_labels, vit_probs, filenames = evaluate_model(
        vit_model, test_loader, "ViT-B/16"
    )
    
    # 4. 评估 ResNet
    resnet_results, resnet_preds, resnet_labels, resnet_probs, _ = evaluate_model(
        resnet_model, test_loader, "ResNet50"
    )
    
    # 5. 生成对比可视化
    plot_comparison(vit_results, resnet_results)
    
    # 6. 生成对比报告
    report = generate_comparison_report(vit_results, resnet_results)
    
    # 7. 保存完整结果
    full_results = {
        'test_dataset': {
            'num_samples': len(test_df),
            'class_distribution': {
                'Evaluation_Board': int(test_df['core_eval_board'].sum()),
                'Power_Management': int(test_df['core_power_management'].sum()),
                'Interface': int(test_df['core_interface'].sum())
            }
        },
        'vit': vit_results,
        'resnet': resnet_results,
        'comparison_summary': report
    }
    
    with open(OUTPUT_DIR / 'full_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 所有结果已保存到: {OUTPUT_DIR}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("评估完成 - 性能总结")
    print("=" * 80)
    print(f"\nViT-B/16:")
    print(f"  Exact Match: {vit_results['exact_match']:.4f}")
    print(f"  Macro-F1:    {vit_results['macro_f1']:.4f}")
    print(f"  Micro-F1:    {vit_results['micro_f1']:.4f}")
    
    print(f"\nResNet50:")
    print(f"  Exact Match: {resnet_results['exact_match']:.4f}")
    print(f"  Macro-F1:    {resnet_results['macro_f1']:.4f}")
    print(f"  Micro-F1:    {resnet_results['micro_f1']:.4f}")
    
    # 判断胜者
    if vit_results['macro_f1'] > resnet_results['macro_f1']:
        winner = "ViT-B/16"
        advantage = vit_results['macro_f1'] - resnet_results['macro_f1']
    else:
        winner = "ResNet50"
        advantage = resnet_results['macro_f1'] - vit_results['macro_f1']
    
    print(f"\n🏆 在 Macro-F1 指标上，**{winner}** 表现更优 (+{advantage:.4f})")
    print("=" * 80)


if __name__ == '__main__':
    main()
