"""
Task 1 来源分类 - ResNet50 vs ViT-B/16 错误案例 Grad-CAM 对比分析
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
import cv2

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handle = None
        
        # Use forward hook and retain_grad instead of backward hook
        self.hook_handle = target_layer.register_forward_hook(self.save_activation)
    
    def save_activation(self, module, input, output):
        # Handle tuple output
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.activations = output
        # Retain gradient for backward pass
        if self.activations.requires_grad is False:
            self.activations.requires_grad_(True)
        if hasattr(self.activations, 'retain_grad'):
            self.activations.retain_grad()
    
    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
    
    def generate_cam(self, input_tensor, class_idx, is_vit=False):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad(set_to_none=True)
        
        # Backward pass
        score = output[0, class_idx]
        score.backward(retain_graph=False)
        
        acts = self.activations
        grads = None if acts is None else acts.grad
        
        if acts is None or grads is None:
            # Fallback to zeros
            return np.zeros((14, 14), dtype=np.float32) if is_vit else np.zeros((7, 7), dtype=np.float32)
        
        # Handle tuple output
        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        if isinstance(grads, (tuple, list)):
            grads = grads[0]
        
        if is_vit:
            # ViT: token-level grad*act contribution (more stable than channel weights)
            # Handle [seq, B, C] or [B, seq, C] format
            if acts.dim() == 3:
                if acts.shape[0] != input_tensor.shape[0] and acts.shape[1] == input_tensor.shape[0]:
                    # [seq, B, C] -> [B, seq, C]
                    acts = acts.permute(1, 0, 2).contiguous()
                    grads = grads.permute(1, 0, 2).contiguous()
                
                # Remove CLS token
                acts_no_cls = acts[:, 1:, :]   # [B, T, C]
                grads_no_cls = grads[:, 1:, :] # [B, T, C]
                
                # Token importance: ReLU(sum_c act * grad)
                token_score = (acts_no_cls * grads_no_cls).sum(dim=2)  # [B, T]
                token_score = torch.relu(token_score)
                
                Ttokens = token_score.shape[1]
                side = int(Ttokens ** 0.5)
                if side * side != Ttokens:
                    # Cannot reshape, use mean
                    cam = token_score[0].detach().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    return cam
                
                cam = token_score[0].reshape(side, side)  # [H, W]
            else:
                # Fallback
                cam = grads.abs().mean(dim=0)
                cam = cam.detach().cpu().numpy()
        else:
            # ResNet: standard Grad-CAM with channel weights
            if acts.dim() != 4:
                # Fallback
                cam = grads.abs().mean(dim=0)
                cam = cam.detach().cpu().numpy()
            else:
                weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                cam = (weights * acts).sum(dim=1)  # [B, H, W]
                cam = torch.relu(cam)[0]
        
        cam = cam.detach().cpu().numpy().astype(np.float32)
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_models():
    """加载ResNet50和ViT-B/16模型"""
    print("\n创建模型架构...")
    
    # ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(resnet.fc.in_features, 5)
    resnet_target = resnet.layer4[-1]
    
    # ViT-B/16
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads.head = nn.Linear(vit.heads.head.in_features, 5)
    # Use ln_1 from last encoder block (more stable token semantics)
    blk = vit.encoder.layers[-1]
    vit_target = blk.ln_1 if hasattr(blk, 'ln_1') else blk
    
    print("✓ ResNet50 架构创建成功")
    print("✓ ViT-B/16 架构创建成功")
    
    return resnet, resnet_target, vit, vit_target


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir).parent
        self.transform = transform
        self.label_map = {'altium': 0, 'eagle': 1, 'jlc': 2, 'kicad': 3, 'orcad': 4}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source = row['source']
        filename = row['filename']
        
        img_path = self.img_dir / 'test' / source / filename
        if not img_path.exists():
            img_path = self.img_dir / source / filename
        
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


def find_common_errors(resnet, vit, test_loader, device, num_errors=8):
    """找出两个模型都预测错误的案例"""
    resnet.eval()
    vit.eval()
    error_cases = []
    
    with torch.no_grad():
        for images, labels, filenames, original_imgs, img_paths, sources in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # ResNet预测
            resnet_outputs = resnet(images)
            resnet_probs = torch.softmax(resnet_outputs, dim=1)
            resnet_preds = resnet_outputs.argmax(dim=1)
            
            # ViT预测
            vit_outputs = vit(images)
            vit_probs = torch.softmax(vit_outputs, dim=1)
            vit_preds = vit_outputs.argmax(dim=1)
            
            # 找出两个模型都错误的样本
            for i in range(len(images)):
                if resnet_preds[i] != labels[i] and vit_preds[i] != labels[i]:
                    error_info = {
                        'filename': filenames[i],
                        'image': images[i],
                        'original_img': original_imgs[i],
                        'img_path': img_paths[i],
                        'label': labels[i].item(),
                        'source': sources[i],
                        'resnet_pred': resnet_preds[i].item(),
                        'resnet_probs': resnet_probs[i].cpu().numpy(),
                        'vit_pred': vit_preds[i].item(),
                        'vit_probs': vit_probs[i].cpu().numpy(),
                    }
                    error_cases.append(error_info)
                    
                    if len(error_cases) >= num_errors:
                        return error_cases
    
    return error_cases


def visualize_comparison(error_case, resnet, resnet_target, vit, vit_target, 
                        class_names, device, output_dir):
    """生成ResNet vs ViT的Grad-CAM对比图"""
    filename = error_case['filename']
    image_tensor = error_case['image'].unsqueeze(0).to(device)
    original_img = error_case['original_img']
    label_idx = error_case['label']
    true_class = class_names[label_idx]
    
    resnet_pred = error_case['resnet_pred']
    resnet_probs = error_case['resnet_probs']
    vit_pred = error_case['vit_pred']
    vit_probs = error_case['vit_probs']
    
    # 准备原始图片
    original_np = np.array(original_img.resize((224, 224)))
    
    # 创建Grad-CAM对象
    resnet_gradcam = GradCAM(resnet, resnet_target)
    vit_gradcam = GradCAM(vit, vit_target)
    
    # 创建对比图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # 第一行：原图 + 信息
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_np)
    ax_orig.axis('off')
    ax_orig.set_title(f'Original Image\n{filename}', fontsize=12, weight='bold')
    
    # 模型预测信息
    ax_info = fig.add_subplot(gs[0, 1:])
    ax_info.axis('off')
    
    info_text = f"Ground Truth: {true_class}\n\n"
    info_text += f"ResNet50 Prediction:\n"
    info_text += f"  Predicted: {class_names[resnet_pred]}\n"
    info_text += f"  Confidence: {resnet_probs[resnet_pred]:.3f}\n"
    info_text += f"  True class conf: {resnet_probs[label_idx]:.3f}\n\n"
    info_text += f"ViT-B/16 Prediction:\n"
    info_text += f"  Predicted: {class_names[vit_pred]}\n"
    info_text += f"  Confidence: {vit_probs[vit_pred]:.3f}\n"
    info_text += f"  True class conf: {vit_probs[label_idx]:.3f}\n"
    
    ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 第二行：ResNet Grad-CAM
    # ResNet预测类
    resnet_cam_pred = resnet_gradcam.generate_cam(image_tensor, resnet_pred, is_vit=False)
    resnet_cam_pred_resized = cv2.resize(resnet_cam_pred, (224, 224))
    resnet_heatmap_pred = cv2.applyColorMap(np.uint8(255 * resnet_cam_pred_resized), cv2.COLORMAP_JET)
    resnet_heatmap_pred = cv2.cvtColor(resnet_heatmap_pred, cv2.COLOR_BGR2RGB)
    resnet_superimposed_pred = resnet_heatmap_pred * 0.4 + original_np * 0.6
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(np.uint8(resnet_superimposed_pred))
    ax1.axis('off')
    ax1.set_title(f'ResNet - Predicted\n{class_names[resnet_pred]} ({resnet_probs[resnet_pred]:.3f})', 
                 fontsize=11, color='red', weight='bold')
    
    # ResNet真实类
    resnet_cam_true = resnet_gradcam.generate_cam(image_tensor, label_idx, is_vit=False)
    resnet_cam_true_resized = cv2.resize(resnet_cam_true, (224, 224))
    resnet_heatmap_true = cv2.applyColorMap(np.uint8(255 * resnet_cam_true_resized), cv2.COLORMAP_JET)
    resnet_heatmap_true = cv2.cvtColor(resnet_heatmap_true, cv2.COLOR_BGR2RGB)
    resnet_superimposed_true = resnet_heatmap_true * 0.4 + original_np * 0.6
    
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(np.uint8(resnet_superimposed_true))
    ax2.axis('off')
    ax2.set_title(f'ResNet - True\n{true_class} ({resnet_probs[label_idx]:.3f})', 
                 fontsize=11, color='green', weight='bold')
    
    # ResNet纯热力图对比
    ax3 = fig.add_subplot(gs[1, 2])
    im1 = ax3.imshow(resnet_cam_pred_resized, cmap='jet')
    ax3.axis('off')
    ax3.set_title('ResNet - Pred Heatmap', fontsize=10)
    plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = fig.add_subplot(gs[1, 3])
    im2 = ax4.imshow(resnet_cam_true_resized, cmap='jet')
    ax4.axis('off')
    ax4.set_title('ResNet - True Heatmap', fontsize=10)
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    
    # 第三行：ViT Grad-CAM
    # ViT预测类
    vit_cam_pred = vit_gradcam.generate_cam(image_tensor, vit_pred, is_vit=True)
    vit_cam_pred_resized = cv2.resize(vit_cam_pred, (224, 224))
    vit_heatmap_pred = cv2.applyColorMap(np.uint8(255 * vit_cam_pred_resized), cv2.COLORMAP_JET)
    vit_heatmap_pred = cv2.cvtColor(vit_heatmap_pred, cv2.COLOR_BGR2RGB)
    vit_superimposed_pred = vit_heatmap_pred * 0.4 + original_np * 0.6
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(np.uint8(vit_superimposed_pred))
    ax5.axis('off')
    ax5.set_title(f'ViT - Predicted\n{class_names[vit_pred]} ({vit_probs[vit_pred]:.3f})', 
                 fontsize=11, color='red', weight='bold')
    
    # ViT真实类
    vit_cam_true = vit_gradcam.generate_cam(image_tensor, label_idx, is_vit=True)
    vit_cam_true_resized = cv2.resize(vit_cam_true, (224, 224))
    vit_heatmap_true = cv2.applyColorMap(np.uint8(255 * vit_cam_true_resized), cv2.COLORMAP_JET)
    vit_heatmap_true = cv2.cvtColor(vit_heatmap_true, cv2.COLOR_BGR2RGB)
    vit_superimposed_true = vit_heatmap_true * 0.4 + original_np * 0.6
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(np.uint8(vit_superimposed_true))
    ax6.axis('off')
    ax6.set_title(f'ViT - True\n{true_class} ({vit_probs[label_idx]:.3f})', 
                 fontsize=11, color='green', weight='bold')
    
    # ViT纯热力图对比
    ax7 = fig.add_subplot(gs[2, 2])
    im3 = ax7.imshow(vit_cam_pred_resized, cmap='jet')
    ax7.axis('off')
    ax7.set_title('ViT - Pred Heatmap', fontsize=10)
    plt.colorbar(im3, ax=ax7, fraction=0.046, pad=0.04)
    
    ax8 = fig.add_subplot(gs[2, 3])
    im4 = ax8.imshow(vit_cam_true_resized, cmap='jet')
    ax8.axis('off')
    ax8.set_title('ViT - True Heatmap', fontsize=10)
    plt.colorbar(im4, ax=ax8, fraction=0.046, pad=0.04)
    
    # 总标题
    plt.suptitle(f'ResNet50 vs ViT-B/16 Grad-CAM Comparison\n' + 
                f'True: {true_class} | ResNet→{class_names[resnet_pred]} | ViT→{class_names[vit_pred]}',
                fontsize=14, weight='bold')
    
    # 保存
    output_path = output_dir / f'comparison_{filename.replace(".png", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")


def main():
    # 配置
    IMAGE_DIR = Path(os.environ.get("TASK1_DATA_ROOT", r'D:\FYP\data\EDA_cls_dataset')) / 'test'
    OUTPUT_DIR = Path('task1_resnet_vit_comparison')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_ERRORS = 8
    CLASS_NAMES = ['Altium', 'Eagle', 'JLC', 'KiCad', 'OrCAD']
    
    print("=" * 80)
    print("Task 1 ResNet50 vs ViT-B/16 错误案例 Grad-CAM 对比分析")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    
    # 加载模型
    print("\n" + "=" * 80)
    print("加载模型...")
    print("=" * 80)
    
    resnet, resnet_target, vit, vit_target = load_models()
    resnet = resnet.to(DEVICE)
    vit = vit.to(DEVICE)
    
    print(f"✓ ResNet target layer: {resnet_target.__class__.__name__}")
    print(f"✓ ViT target layer: {vit_target.__class__.__name__}")
    
    # 加载数据集
    print("\n" + "=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    # 直接从文件夹读取
    image_files = []
    sources = []
    for source_folder in ['altium', 'eagle', 'jlc', 'kicad', 'orcad']:
        source_path = IMAGE_DIR / source_folder
        if source_path.exists():
            for img_file in source_path.glob('*.png'):
                image_files.append(img_file.name)
                sources.append(source_folder)
    
    test_df = pd.DataFrame({'filename': image_files, 'source': sources})
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
    
    # 找出两个模型都错误的案例
    print("\n" + "=" * 80)
    print(f"查找两个模型都预测错误的案例 (前{NUM_ERRORS}个)...")
    print("=" * 80)
    
    error_cases = find_common_errors(resnet, vit, test_loader, DEVICE, NUM_ERRORS)
    print(f"\n找到 {len(error_cases)} 个两个模型都错误的案例")
    
    if len(error_cases) == 0:
        print("没有发现共同错误案例！")
        return
    
    # 生成对比可视化
    print("\n" + "=" * 80)
    print("生成 ResNet vs ViT Grad-CAM 对比可视化...")
    print("=" * 80)
    
    for i, error_case in enumerate(error_cases, 1):
        true_class = CLASS_NAMES[error_case['label']]
        resnet_pred_class = CLASS_NAMES[error_case['resnet_pred']]
        vit_pred_class = CLASS_NAMES[error_case['vit_pred']]
        
        print(f"\n处理 {i}/{len(error_cases)}: {error_case['filename']}")
        print(f"  True: {true_class}")
        print(f"  ResNet→{resnet_pred_class}, ViT→{vit_pred_class}")
        
        visualize_comparison(
            error_case, resnet, resnet_target, vit, vit_target,
            CLASS_NAMES, DEVICE, OUTPUT_DIR
        )
    
    # 生成汇总报告
    print("\n" + "=" * 80)
    print("生成汇总报告...")
    print("=" * 80)
    
    report_path = OUTPUT_DIR / 'comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Task 1 ResNet50 vs ViT-B/16 错误案例对比分析\n")
        f.write("=" * 80 + "\n\n")
        
        for i, case in enumerate(error_cases, 1):
            true_class = CLASS_NAMES[case['label']]
            resnet_pred = CLASS_NAMES[case['resnet_pred']]
            vit_pred = CLASS_NAMES[case['vit_pred']]
            
            f.write(f"{i}. {case['filename']}\n")
            f.write(f"   Ground Truth: {true_class}\n")
            f.write(f"   ResNet50: {resnet_pred} (conf: {case['resnet_probs'][case['resnet_pred']]:.3f})\n")
            f.write(f"   ViT-B/16: {vit_pred} (conf: {case['vit_probs'][case['vit_pred']]:.3f})\n")
            f.write(f"   Path: {case['img_path']}\n\n")
    
    print(f"✓ 保存: comparison_report.txt")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n生成的文件保存在: {OUTPUT_DIR}")
    print(f"  - {len(error_cases)} 张 ResNet vs ViT 对比图")
    print(f"  - 1 个汇总报告文件")


if __name__ == '__main__':
    main()
