"""
Task 1 来源分类错误案例的 Grad-CAM 可视化分析
找出分类错误的样本，生成Grad-CAM热力图，分析模型关注区域
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass for target class
        target = output[:, class_idx]
        target.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def load_model_and_data(model_name='convnext_tiny'):
    """加载指定模型"""
    print(f"\n加载模型: {model_name}")
    
    # 查找模型文件
    # 假设使用最新训练的模型
    if model_name == 'convnext_tiny':
        from torchvision.models import convnext_tiny
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 5)
        target_layer = model.features[-1][-1]  # 最后一个block
    elif model_name == 'vit_b_16':
        from torchvision.models import vit_b_16
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 5)
        target_layer = model.encoder.layers[-1].ln_1
    else:  # resnet50
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        target_layer = model.layer4[-1]  # 最后一个residual block
    
    print(f"✓ 模型架构创建成功")
    
    return model, target_layer, model_name


class ImageDataset(Dataset):
    """图像数据集"""
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir).parent  # 回到上一级，因为source是子文件夹
        self.transform = transform
        self.label_map = {'altium': 0, 'eagle': 1, 'jlc': 2, 'kicad': 3, 'orcad': 4}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 图片路径
        source = row['source']
        filename = row['filename']
        # 尝试test文件夹
        img_path = self.img_dir / 'test' / source / filename
        if not img_path.exists():
            # 尝试根目录
            img_path = self.img_dir / source / filename
        
        # 读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        original_img = image.copy()
        
        if self.transform:
            image = self.transform(image)
        
        label = self.label_map[source.lower()]
        
        return image, label, filename, original_img, str(img_path), source


def find_error_cases(model, test_loader, device, num_errors=10):
    """找出分类错误的案例"""
    model.eval()
    error_cases = []
    
    with torch.no_grad():
        for images, labels, filenames, original_imgs, img_paths, sources in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # 找出错误的样本
            for i in range(len(images)):
                if preds[i] != labels[i]:
                    error_info = {
                        'filename': filenames[i],
                        'image': images[i],
                        'original_img': original_imgs[i],
                        'img_path': img_paths[i],
                        'pred': preds[i].item(),
                        'label': labels[i].item(),
                        'probs': probs[i].cpu().numpy(),
                        'source': sources[i]
                    }
                    error_cases.append(error_info)
                    
                    if len(error_cases) >= num_errors:
                        return error_cases
    
    return error_cases


def visualize_error_with_gradcam(error_case, model, target_layer, class_names, device, output_dir):
    """为错误案例生成Grad-CAM可视化"""
    filename = error_case['filename']
    image_tensor = error_case['image'].unsqueeze(0).to(device)
    original_img = error_case['original_img']
    pred_idx = error_case['pred']
    label_idx = error_case['label']
    probs = error_case['probs']
    true_source = error_case['source']
    
    pred_class = class_names[pred_idx]
    true_class = class_names[label_idx]
    
    # 创建Grad-CAM对象
    grad_cam = GradCAM(model, target_layer)
    
    # 准备原始图片用于叠加
    original_np = np.array(original_img.resize((224, 224)))
    
    # 为预测类和真实类生成CAM
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
    
    # 第一行：原图 + 预测类CAM + 真实类CAM + Top-3预测
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_np)
    ax_orig.axis('off')
    ax_orig.set_title(f'Original Image\n{filename}', fontsize=11, weight='bold')
    
    # 预测类的CAM
    cam_pred = grad_cam.generate_cam(image_tensor, pred_idx)
    cam_pred_resized = cv2.resize(cam_pred, (224, 224))
    heatmap_pred = cv2.applyColorMap(np.uint8(255 * cam_pred_resized), cv2.COLORMAP_JET)
    heatmap_pred = cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB)
    superimposed_pred = heatmap_pred * 0.4 + original_np * 0.6
    
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_pred.imshow(np.uint8(superimposed_pred))
    ax_pred.axis('off')
    ax_pred.set_title(f'Predicted: {pred_class}\n(Confidence: {probs[pred_idx]:.3f})', 
                     fontsize=11, color='red', weight='bold')
    
    # 真实类的CAM
    cam_true = grad_cam.generate_cam(image_tensor, label_idx)
    cam_true_resized = cv2.resize(cam_true, (224, 224))
    heatmap_true = cv2.applyColorMap(np.uint8(255 * cam_true_resized), cv2.COLORMAP_JET)
    heatmap_true = cv2.cvtColor(heatmap_true, cv2.COLOR_BGR2RGB)
    superimposed_true = heatmap_true * 0.4 + original_np * 0.6
    
    ax_true = fig.add_subplot(gs[0, 2])
    ax_true.imshow(np.uint8(superimposed_true))
    ax_true.axis('off')
    ax_true.set_title(f'True: {true_class}\n(Confidence: {probs[label_idx]:.3f})', 
                     fontsize=11, color='green', weight='bold')
    
    # Top-3预测概率
    top3_indices = np.argsort(probs)[-3:][::-1]
    ax_bar = fig.add_subplot(gs[0, 3:])
    
    colors = []
    for idx in top3_indices:
        if idx == pred_idx:
            colors.append('red')
        elif idx == label_idx:
            colors.append('green')
        else:
            colors.append('gray')
    
    bars = ax_bar.barh([class_names[i] for i in top3_indices], 
                       [probs[i] for i in top3_indices],
                       color=colors)
    ax_bar.set_xlabel('Confidence', fontsize=11)
    ax_bar.set_title('Top-3 Predictions', fontsize=12, weight='bold')
    ax_bar.set_xlim([0, 1])
    ax_bar.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (idx, bar) in enumerate(zip(top3_indices, bars)):
        width = bar.get_width()
        ax_bar.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{probs[idx]:.3f}',
                   ha='left', va='center', fontsize=10, weight='bold')
    
    # 第二行：所有5个类别的纯热力图
    for class_idx in range(5):
        cam = grad_cam.generate_cam(image_tensor, class_idx)
        cam_resized = cv2.resize(cam, (224, 224))
        
        ax = fig.add_subplot(gs[1, class_idx])
        im = ax.imshow(cam_resized, cmap='jet')
        ax.axis('off')
        
        # 标题颜色
        if class_idx == pred_idx:
            color = 'red'
            weight = 'bold'
        elif class_idx == label_idx:
            color = 'green'
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'
        
        ax.set_title(f'{class_names[class_idx]}\n({probs[class_idx]:.3f})', 
                    fontsize=10, color=color, weight=weight)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 总标题
    plt.suptitle(f'Task 1 Source Classification - Error Analysis\n' + 
                f'Predicted: {pred_class} | True: {true_class} | File: {filename}',
                fontsize=14, weight='bold')
    
    # 保存
    output_path = output_dir / f'task1_gradcam_error_{filename.replace(".png", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")


def main():
    # 配置
    DATASET_CSV = Path('task1_source_classification/metrics/dataset_statistics.csv')
    IMAGE_DIR = Path(os.environ.get("TASK1_DATA_ROOT", r'D:\FYP\data\EDA_cls_dataset')) / 'test'  # 使用test文件夹
    OUTPUT_DIR = Path('task1_gradcam_error_analysis')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_ERRORS = 10
    CLASS_NAMES = ['Altium', 'Eagle', 'JLC', 'KiCad', 'OrCAD']
    MODEL_NAME = 'convnext_tiny'  # 可选: convnext_tiny, resnet50, vit_b_16
    
    print("=" * 80)
    print("Task 1 来源分类错误案例 Grad-CAM 分析")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    
    # 加载模型（仅架构，用于演示Grad-CAM）
    print("\n" + "=" * 80)
    print("创建模型架构...")
    print("=" * 80)
    
    model, target_layer, model_name = load_model_and_data(MODEL_NAME)
    
    # 使用预训练权重进行演示（实际应该加载训练好的权重）
    print("\n注意: 使用预训练权重进行演示，实际应该加载Task 1训练好的权重")
    
    if MODEL_NAME == 'convnext_tiny':
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        pretrained_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # 复制除了最后一层的所有权重
        model.features.load_state_dict(pretrained_model.features.state_dict())
        model.classifier[0].load_state_dict(pretrained_model.classifier[0].state_dict())
        model.classifier[1].load_state_dict(pretrained_model.classifier[1].state_dict())
    elif MODEL_NAME == 'resnet50':
        pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 复制除了fc的所有权重
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    model = model.to(DEVICE)
    print(f"✓ Target layer: {target_layer.__class__.__name__}")
    
    # 加载数据集
    print("\n" + "=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    # 直接从文件夹读取数据
    image_files = []
    sources = []
    for source_folder in ['altium', 'eagle', 'jlc', 'kicad', 'orcad']:
        source_path = IMAGE_DIR / source_folder
        if source_path.exists():
            for img_file in source_path.glob('*.png'):
                image_files.append(img_file.name)
                sources.append(source_folder)
    
    # 创建DataFrame
    test_df = pd.DataFrame({
        'filename': image_files,
        'source': sources
    })
    
    # 随机采样一部分用于演示（加快速度）
    test_df = test_df.sample(n=min(200, len(test_df)), random_state=42).reset_index(drop=True)
    print(f"使用采样数据集: {len(test_df)} 样本")
    
    print(f"\n数据分布:")
    for source in CLASS_NAMES:
        count = (test_df['source'].str.lower() == source.lower()).sum()
        print(f"  {source}: {count}")
    
    # 创建数据加载器
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageDataset(test_df, IMAGE_DIR, test_transform)
    
    def custom_collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        filenames = [item[2] for item in batch]
        original_imgs = [item[3] for item in batch]
        img_paths = [item[4] for item in batch]
        sources = [item[5] for item in batch]
        return images, labels, filenames, original_imgs, img_paths, sources
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, 
                            num_workers=0, collate_fn=custom_collate)
    
    # 找出错误案例
    print("\n" + "=" * 80)
    print(f"查找错误案例 (前{NUM_ERRORS}个)...")
    print("=" * 80)
    
    error_cases = find_error_cases(model, test_loader, DEVICE, NUM_ERRORS)
    print(f"\n找到 {len(error_cases)} 个错误案例")
    
    if len(error_cases) == 0:
        print("没有发现错误案例！模型表现完美。")
        return
    
    # 统计混淆矩阵
    confusion = {true: {pred: 0 for pred in CLASS_NAMES} for true in CLASS_NAMES}
    for case in error_cases:
        true_class = CLASS_NAMES[case['label']]
        pred_class = CLASS_NAMES[case['pred']]
        confusion[true_class][pred_class] += 1
    
    print("\n错误案例混淆统计:")
    print(f"{'True/Pred':<10}", end='')
    for pred in CLASS_NAMES:
        print(f"{pred:<10}", end='')
    print()
    print("-" * 60)
    for true in CLASS_NAMES:
        print(f"{true:<10}", end='')
        for pred in CLASS_NAMES:
            count = confusion[true][pred]
            if count > 0:
                print(f"{count:<10}", end='')
            else:
                print(f"{'.':<10}", end='')
        print()
    
    # 生成Grad-CAM可视化
    print("\n" + "=" * 80)
    print("生成 Grad-CAM 可视化...")
    print("=" * 80)
    
    for i, error_case in enumerate(error_cases, 1):
        pred_class = CLASS_NAMES[error_case['pred']]
        true_class = CLASS_NAMES[error_case['label']]
        print(f"\n处理 {i}/{len(error_cases)}: {error_case['filename']}")
        print(f"  Predicted: {pred_class} | True: {true_class}")
        
        visualize_error_with_gradcam(
            error_case, model, target_layer, CLASS_NAMES, DEVICE, OUTPUT_DIR
        )
    
    # 生成汇总统计
    print("\n" + "=" * 80)
    print("生成错误分析汇总...")
    print("=" * 80)
    
    # 错误案例列表
    error_list_path = OUTPUT_DIR / 'task1_error_cases_list.txt'
    with open(error_list_path, 'w', encoding='utf-8') as f:
        f.write("Task 1 来源分类错误案例列表\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n\n")
        
        for i, case in enumerate(error_cases, 1):
            pred_class = CLASS_NAMES[case['pred']]
            true_class = CLASS_NAMES[case['label']]
            f.write(f"{i}. {case['filename']}\n")
            f.write(f"   Path: {case['img_path']}\n")
            f.write(f"   Predicted: {pred_class} (conf: {case['probs'][case['pred']]:.3f})\n")
            f.write(f"   True:      {true_class} (conf: {case['probs'][case['label']]:.3f})\n")
            f.write(f"   All probs: {', '.join([f'{CLASS_NAMES[i]}={case['probs'][i]:.3f}' for i in range(5)])}\n")
            f.write("\n")
    
    print(f"✓ 保存: task1_error_cases_list.txt")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n生成的文件保存在: {OUTPUT_DIR}")
    print(f"  - {len(error_cases)} 张 Grad-CAM 错误分析图")
    print(f"  - 1 个错误案例列表文件")


if __name__ == '__main__':
    main()
