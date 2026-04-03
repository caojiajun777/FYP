"""
完整的输入区域消融实验 - ResNet50 & ConvNeXt

测试模型对不同图像区域的依赖程度：
- Full: 完整图片（baseline）
- Bottom Only: 只保留底部区域（标题栏）
- Center Only: 只保留中心区域（核心电路）
- Footer Masked: 完整图片但遮罩页脚区域

目标：定量分析模型对不同区域的依赖程度，比较ResNet50和ConvNeXt的区别
"""

import os
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import json
import argparse

# 配置
CLASSES = ["altium", "eagle", "jlc", "kicad", "orcad"]
TEST_DATA_ROOT = Path(os.environ.get("TASK1_KFOLD_DATA_ROOT", r"D:\FYP\data\EDA_cls_dataset_kfold"))
OUTPUT_DIR = Path(os.environ.get("TASK1_OUTPUT_DIR", r"D:\FYP\Classifier\paper_results\ablation_studies\input_regions"))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 区域定义（基于224x224）
REGIONS = {
    'bottom': (0, 168, 224, 56),      # 底部1/4（标题栏）
    'center': (56, 56, 112, 112),     # 中心区域（核心电路）
    'footer': (0, 168, 224, 56),      # 页脚区域（与bottom相同位置，但用于遮罩）
    'full': None                       # 完整图片
}

# 模型路径配置
MODEL_PATHS = {
    'resnet50': Path(os.environ.get("KFOLD_RESNET_MODEL_PATH", r"D:\FYP\runs_kfold\resnet50\fold0\best_model.pt")),
    'convnext': Path(os.environ.get("KFOLD_CONVNEXT_MODEL_PATH", r"D:\FYP\runs_kfold\convnext_tiny\fold0\best_model.pt")),
}

class RegionMaskTransform:
    """在特定区域外应用遮罩"""
    def __init__(self, region_type='full', fill_value=0.5):
        self.region_type = region_type
        self.fill_value = fill_value
        
    def __call__(self, img):
        """
        img: PIL Image (224x224)
        返回: 应用区域遮罩后的tensor
        """
        # 转为tensor
        img_tensor = T.ToTensor()(img)
        
        if self.region_type == 'full':
            # 完整图片，不遮罩
            pass
        elif self.region_type == 'bottom':
            # 只保留底部区域，其它填灰
            x, y, w, h = REGIONS['bottom']
            mask = torch.ones_like(img_tensor) * self.fill_value
            mask[:, y:y+h, x:x+w] = img_tensor[:, y:y+h, x:x+w]
            img_tensor = mask
        elif self.region_type == 'center':
            # 只保留中心区域，其它填灰
            x, y, w, h = REGIONS['center']
            mask = torch.ones_like(img_tensor) * self.fill_value
            mask[:, y:y+h, x:x+w] = img_tensor[:, y:y+h, x:x+w]
            img_tensor = mask
        elif self.region_type == 'footer_masked':
            # 完整图片，但遮罩页脚区域
            x, y, w, h = REGIONS['footer']
            img_tensor[:, y:y+h, x:x+w] = self.fill_value
        
        # 归一化
        normalize = T.Normalize(MEAN, STD)
        img_tensor = normalize(img_tensor)
        
        return img_tensor

def load_model(model_type):
    """加载指定类型的模型"""
    model_path = MODEL_PATHS[model_type]
    print(f"📦 Loading {model_type.upper()} from {model_path}...")
    
    if model_type == 'resnet50':
        model = tv.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    elif model_type == 'convnext':
        model = tv.models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"  ✅ Model loaded")
    return model

def create_dataloader(region_type, batch_size=64):
    """创建指定区域遮罩的数据加载器"""
    transform = T.Compose([
        T.Resize((224, 224)),
        RegionMaskTransform(region_type=region_type, fill_value=0.5),
    ])
    
    dataset = ImageFolder(TEST_DATA_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return loader, dataset

def evaluate_region(model, loader, model_name, region_name):
    """评估模型在特定区域配置下的性能"""
    print(f"\n🧪 Evaluating {model_name.upper()} on {region_name.upper()} region...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  Testing {region_name}", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 每类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(CLASSES))
    )
    
    results = {
        'region': region_name,
        'model': model_name,
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            CLASSES[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(CLASSES))
        },
        'total_samples': len(all_labels)
    }
    
    print(f"  ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✅ Macro F1: {macro_f1:.4f}")
    print(f"  ✅ Weighted F1: {weighted_f1:.4f}")
    
    return results

def generate_comparison_report(results_dict):
    """生成完整的对比分析报告"""
    print(f"\n📝 Generating comprehensive comparison report...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "complete_region_ablation_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 完整输入区域消融实验报告 - ResNet50 vs ConvNeXt\n\n")
        f.write("## 1. 实验目的\n\n")
        f.write("通过系统性消融实验，回答两个核心问题：\n\n")
        f.write("1. **ResNet50和ConvNeXt对图像不同区域的依赖程度如何？**\n")
        f.write("2. **两种架构在特征提取策略上有何差异？**\n\n")
        
        f.write("## 2. 实验设计\n\n")
        f.write("### 2.1 区域定义\n\n")
        f.write("基于224×224输入图像：\n\n")
        f.write("| 区域配置 | 操作 | 包含内容 | 测试目的 |\n")
        f.write("|---------|------|----------|----------|\n")
        f.write("| **Full** | 完整图片 | 全部信息 | Baseline性能 |\n")
        f.write("| **Bottom Only** | 保留底部1/4，其它填灰 | 标题栏、LOGO、版本信息 | 标题栏判别力 |\n")
        f.write("| **Center Only** | 保留中心1/4，其它填灰 | 核心电路、主要器件 | 电路区域判别力 |\n")
        f.write("| **Footer Masked** | 完整图片，遮罩底部1/4 | 去除标题栏的完整图 | 非标题栏信息充分性 |\n\n")
        
        f.write("### 2.2 模型配置\n\n")
        f.write("| 模型 | 参数量 | 架构特点 | 模型路径 |\n")
        f.write("|------|--------|----------|----------|\n")
        f.write("| ResNet50 | 25.6M | 残差连接 | fold0/best_model.pt |\n")
        f.write("| ConvNeXt-Tiny | 27.8M | 现代化卷积 | fold0/best_model.pt |\n\n")
        
        f.write("## 3. 实验结果\n\n")
        
        # 为每个模型生成结果表
        for model_name in ['resnet50', 'convnext']:
            if model_name not in results_dict:
                continue
                
            f.write(f"### 3.{1 if model_name == 'resnet50' else 2} {model_name.upper()} 结果\n\n")
            
            model_results = results_dict[model_name]
            
            # 提取各配置的指标
            full_acc = model_results['full']['accuracy']
            bottom_acc = model_results['bottom']['accuracy']
            center_acc = model_results['center']['accuracy']
            footer_acc = model_results['footer_masked']['accuracy']
            
            full_f1 = model_results['full']['macro_f1']
            bottom_f1 = model_results['bottom']['macro_f1']
            center_f1 = model_results['center']['macro_f1']
            footer_f1 = model_results['footer_masked']['macro_f1']
            
            f.write("| 区域配置 | Accuracy | Macro F1 | Weighted F1 | vs Full (Acc) | vs Full (F1) |\n")
            f.write("|---------|----------|----------|-------------|---------------|-------------|\n")
            f.write(f"| **Full** | **{full_acc:.4f}** | **{full_f1:.4f}** | {model_results['full']['weighted_f1']:.4f} | - | - |\n")
            f.write(f"| Bottom Only | {bottom_acc:.4f} | {bottom_f1:.4f} | {model_results['bottom']['weighted_f1']:.4f} | {(bottom_acc - full_acc):.4f} ({(bottom_acc/full_acc - 1)*100:+.2f}%) | {(bottom_f1 - full_f1):.4f} ({(bottom_f1/full_f1 - 1)*100:+.2f}%) |\n")
            f.write(f"| Center Only | {center_acc:.4f} | {center_f1:.4f} | {model_results['center']['weighted_f1']:.4f} | {(center_acc - full_acc):.4f} ({(center_acc/full_acc - 1)*100:+.2f}%) | {(center_f1 - full_f1):.4f} ({(center_f1/full_f1 - 1)*100:+.2f}%) |\n")
            f.write(f"| Footer Masked | {footer_acc:.4f} | {footer_f1:.4f} | {model_results['footer_masked']['weighted_f1']:.4f} | {(footer_acc - full_acc):.4f} ({(footer_acc/full_acc - 1)*100:+.2f}%) | {(footer_f1 - full_f1):.4f} ({(footer_f1/full_f1 - 1)*100:+.2f}%) |\n\n")
            
            # 每类别F1对比
            f.write(f"#### 每类别F1分数\n\n")
            f.write("| 类别 | Full | Bottom | Center | Footer Masked | Bottom保留率 | Center保留率 |\n")
            f.write("|------|------|--------|--------|---------------|--------------|-------------|\n")
            
            for cls in CLASSES:
                full_cls_f1 = model_results['full']['per_class'][cls]['f1']
                bottom_cls_f1 = model_results['bottom']['per_class'][cls]['f1']
                center_cls_f1 = model_results['center']['per_class'][cls]['f1']
                footer_cls_f1 = model_results['footer_masked']['per_class'][cls]['f1']
                
                bottom_retention = (bottom_cls_f1 / full_cls_f1) * 100 if full_cls_f1 > 0 else 0
                center_retention = (center_cls_f1 / full_cls_f1) * 100 if full_cls_f1 > 0 else 0
                
                f.write(f"| {cls.upper()} | {full_cls_f1:.4f} | {bottom_cls_f1:.4f} | {center_cls_f1:.4f} | {footer_cls_f1:.4f} | {bottom_retention:.1f}% | {center_retention:.1f}% |\n")
            f.write("\n")
        
        # 模型间对比
        if 'resnet50' in results_dict and 'convnext' in results_dict:
            f.write("## 4. 模型间对比分析\n\n")
            f.write("### 4.1 整体性能对比\n\n")
            f.write("| 区域配置 | ResNet50 Acc | ConvNeXt Acc | 差异 | ResNet50 F1 | ConvNeXt F1 | 差异 |\n")
            f.write("|---------|--------------|--------------|------|-------------|-------------|------|\n")
            
            for region in ['full', 'bottom', 'center', 'footer_masked']:
                region_name = region.replace('_', ' ').title()
                r_acc = results_dict['resnet50'][region]['accuracy']
                c_acc = results_dict['convnext'][region]['accuracy']
                r_f1 = results_dict['resnet50'][region]['macro_f1']
                c_f1 = results_dict['convnext'][region]['macro_f1']
                
                acc_diff = c_acc - r_acc
                f1_diff = c_f1 - r_f1
                
                f.write(f"| {region_name} | {r_acc:.4f} | {c_acc:.4f} | {acc_diff:+.4f} | {r_f1:.4f} | {c_f1:.4f} | {f1_diff:+.4f} |\n")
            f.write("\n")
            
            f.write("### 4.2 区域依赖度对比\n\n")
            
            # 计算各模型的标题栏依赖度
            r_bottom_retention = (results_dict['resnet50']['bottom']['accuracy'] / 
                                results_dict['resnet50']['full']['accuracy']) * 100
            c_bottom_retention = (results_dict['convnext']['bottom']['accuracy'] / 
                                results_dict['convnext']['full']['accuracy']) * 100
            
            r_center_retention = (results_dict['resnet50']['center']['accuracy'] / 
                                results_dict['resnet50']['full']['accuracy']) * 100
            c_center_retention = (results_dict['convnext']['center']['accuracy'] / 
                                results_dict['convnext']['full']['accuracy']) * 100
            
            f.write("| 模型 | Bottom保留率 | Center保留率 | 标题栏依赖度 |\n")
            f.write("|------|--------------|--------------|-------------|\n")
            f.write(f"| ResNet50 | {r_bottom_retention:.1f}% | {r_center_retention:.1f}% | {r_bottom_retention/r_center_retention:.2f}× |\n")
            f.write(f"| ConvNeXt | {c_bottom_retention:.1f}% | {c_center_retention:.1f}% | {c_bottom_retention/c_center_retention:.2f}× |\n\n")
        
        f.write("## 5. 关键发现\n\n")
        
        # 为每个模型总结发现
        for model_name in ['resnet50', 'convnext']:
            if model_name not in results_dict:
                continue
                
            model_results = results_dict[model_name]
            bottom_retention = (model_results['bottom']['accuracy'] / 
                              model_results['full']['accuracy']) * 100
            center_retention = (model_results['center']['accuracy'] / 
                              model_results['full']['accuracy']) * 100
            footer_impact = ((model_results['full']['accuracy'] - 
                            model_results['footer_masked']['accuracy']) / 
                           model_results['full']['accuracy']) * 100
            
            f.write(f"### 5.{1 if model_name == 'resnet50' else 2} {model_name.upper()} 关键发现\n\n")
            f.write(f"1. **标题栏依赖度**: {bottom_retention:.1f}% (Bottom Only保留率)\n")
            f.write(f"2. **电路区域判别力**: {center_retention:.1f}% (Center Only保留率)\n")
            f.write(f"3. **标题栏贡献**: {footer_impact:.1f}% (遮罩Footer后的性能下降)\n")
            f.write(f"4. **标题栏vs电路**: 标题栏重要性是电路的 {bottom_retention/center_retention:.2f}倍\n\n")
        
        f.write("## 6. 论文写作建议\n\n")
        f.write("### 6.1 表格建议\n\n")
        f.write("**表7.X：ResNet50和ConvNeXt的区域消融实验结果**\n\n")
        f.write("| 模型 | Full | Bottom | Center | Footer Masked |\n")
        f.write("|------|------|--------|--------|---------------|\n")
        if 'resnet50' in results_dict:
            r = results_dict['resnet50']
            f.write(f"| ResNet50 | {r['full']['accuracy']:.2%} | {r['bottom']['accuracy']:.2%} | {r['center']['accuracy']:.2%} | {r['footer_masked']['accuracy']:.2%} |\n")
        if 'convnext' in results_dict:
            c = results_dict['convnext']
            f.write(f"| ConvNeXt | {c['full']['accuracy']:.2%} | {c['bottom']['accuracy']:.2%} | {c['center']['accuracy']:.2%} | {c['footer_masked']['accuracy']:.2%} |\n")
        f.write("\n")
        
        f.write("### 6.2 讨论要点\n\n")
        f.write("1. **两种架构的共同点**：都高度依赖标题栏信息\n")
        f.write("2. **性能差异**：ConvNeXt在所有配置下是否优于ResNet50？\n")
        f.write("3. **鲁棒性对比**：Footer Masked配置下谁的性能下降更少？\n")
        f.write("4. **架构启示**：现代化卷积（ConvNeXt）vs传统残差（ResNet）的特征提取差异\n\n")
        
        f.write("## 7. 结论\n\n")
        f.write("本实验通过系统性的区域消融，定量分析了ResNet50和ConvNeXt对不同图像区域的依赖程度，")
        f.write("为理解这两种架构在EDA工具分类任务中的决策机制提供了实证支持。\n\n")
        
        f.write("---\n\n")
        f.write("**实验配置**：\n")
        f.write(f"- 测试集大小：{results_dict[list(results_dict.keys())[0]]['full']['total_samples']} 样本\n")
        f.write(f"- 遮罩填充值：0.5 (灰色)\n")
        f.write("- 评估指标：Accuracy, Macro F1, Weighted F1, Per-class F1\n")
    
    print(f"  ✅ Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='完整的输入区域消融实验')
    parser.add_argument('--models', nargs='+', choices=['resnet50', 'convnext', 'all'],
                       default=['all'], help='要测试的模型')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    args = parser.parse_args()
    
    # 确定要测试的模型
    if 'all' in args.models:
        models_to_test = ['resnet50', 'convnext']
    else:
        models_to_test = args.models
    
    print("\n" + "="*70)
    print("🧪 完整输入区域消融实验 - ResNet50 & ConvNeXt")
    print("="*70)
    print(f"📋 测试模型: {', '.join([m.upper() for m in models_to_test])}")
    print(f"📋 区域配置: Full, Bottom Only, Center Only, Footer Masked")
    print("="*70)
    
    # 存储所有结果
    all_results = {}
    
    # 对每个模型进行测试
    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"🔬 Testing {model_name.upper()}")
        print(f"{'='*70}")
        
        # 加载模型
        model = load_model(model_name)
        
        # 存储该模型的结果
        model_results = {}
        
        # 测试各种区域配置
        regions = ['full', 'bottom', 'center', 'footer_masked']
        region_names = ['Full (Baseline)', 'Bottom Only', 'Center Only', 'Footer Masked']
        
        for region, region_display in zip(regions, region_names):
            print(f"\n{'-'*70}")
            print(f"📍 配置: {region_display}")
            print(f"{'-'*70}")
            
            loader, dataset = create_dataloader(region, batch_size=args.batch_size)
            results = evaluate_region(model, loader, model_name, region)
            model_results[region] = results
        
        all_results[model_name] = model_results
    
    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "complete_region_ablation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {results_path}")
    
    # 生成对比报告
    generate_comparison_report(all_results)
    
    # 打印摘要
    print("\n" + "="*70)
    print("📊 实验结果摘要")
    print("="*70)
    
    for model_name in models_to_test:
        print(f"\n{model_name.upper()}:")
        print(f"{'配置':<20} {'Accuracy':<12} {'Macro F1':<12} {'vs Full':<15}")
        print("-" * 70)
        
        full_acc = all_results[model_name]['full']['accuracy']
        for region in ['full', 'bottom', 'center', 'footer_masked']:
            region_display = region.replace('_', ' ').title()
            acc = all_results[model_name][region]['accuracy']
            f1 = all_results[model_name][region]['macro_f1']
            diff = acc - full_acc
            print(f"{region_display:<20} {acc:.4f} ({acc*100:>5.2f}%) {f1:.4f} ({f1*100:>5.2f}%) {diff:+.4f} ({(acc/full_acc-1)*100:>+6.2f}%)")
    
    print("\n" + "="*70)
    print("✅ 完整输入区域消融实验完成！")
    print(f"📁 结果保存至: {OUTPUT_DIR}/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
