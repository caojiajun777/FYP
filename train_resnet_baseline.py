import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, classification_report
import warnings
warnings.filterwarnings('ignore')

CLASSES = ['power', 'interface', 'communication', 'signal', 'control']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def get_linux_path(base_dir, win_path, filename):
    """将 Windows 中的绝对路径智能转换为 Linux 上的相对图片路径"""
    parts = win_path.replace('\\', '/').split('/')
    if len(parts) >= 2:
        rel_path = f"{parts[-2]}/{parts[-1]}"
        linux_path = os.path.join(base_dir, rel_path)
        if os.path.exists(linux_path):
            return linux_path
    
    # 暴力降级：假设图片直接放在 base_dir 或者它的直接子目录下
    fallback_path = os.path.join(base_dir, filename)
    if os.path.exists(fallback_path):
        return fallback_path
        
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
            
    return os.path.join(base_dir, filename) # 兜底

class EDAMultiLabelDataset(Dataset):
    def __init__(self, json_file, img_base_dir, transform=None):
        self.img_base_dir = img_base_dir
        self.transform = transform
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = get_linux_path(self.img_base_dir, item.get('image_path', ''), item['filename'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            
        if self.transform:
            image = self.transform(image)
            
        # Multi-hot encoding
        labels = item.get('gold_labels', [])
        target = torch.zeros(len(CLASSES), dtype=torch.float32)
        for lbl in labels:
            if lbl in CLASS_TO_IDX:
                target[CLASS_TO_IDX[lbl]] = 1.0
                
        return image, target

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    micro_f1 = f1_score(all_targets, all_preds, average='micro')
    hm_loss = hamming_loss(all_targets, all_preds)
    
    return macro_f1, micro_f1, hm_loss, all_targets, all_preds

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_tf, test_tf = get_transforms()
    
    train_set = EDAMultiLabelDataset(args.train_json, args.img_dir, transform=train_tf)
    val_set = EDAMultiLabelDataset(args.val_json, args.img_dir, transform=test_tf)
    test_set = EDAMultiLabelDataset(args.test_json, args.img_dir, transform=test_tf)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_macro = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step()
        
        val_macro, val_micro, val_ham, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Macro-F1: {val_macro:.4f} | Val Micro-F1: {val_micro:.4f} | Hamming: {val_ham:.4f}")
        
        if val_macro > best_macro:
            best_macro = val_macro
            torch.save(model.state_dict(), 'best_resnet50_baseline.pth')
            print(">>> Saved optimal model")
            
    print("\n========= Final Testing Protocol =========")
    model.load_state_dict(torch.load('best_resnet50_baseline.pth'))
    test_macro, test_micro, test_ham, y_true, y_pred = evaluate(model, test_loader, device)
    
    print(f"Test Macro-F1: {test_macro:.4f}")
    print(f"Test Micro-F1: {test_micro:.4f}")
    print(f"Test Hamming Loss: {test_ham:.4f}")
    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help="Path to EDA_cls_dataset_full on cloud")
    parser.add_argument('--train_json', type=str, default='task2_function_prediction/train_sets/train_high.json')
    parser.add_argument('--val_json', type=str, default='task2_function_prediction/gold_standard/val_split.json')
    parser.add_argument('--test_json', type=str, default='task2_function_prediction/gold_standard/test_split.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    args = parser.parse_args()
    
    train_model(args)
