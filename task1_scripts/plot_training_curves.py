"""
从训练历史中读取数据并重新绘制训练曲线
将 train_loss 和 val_loss 画在同一张图上
"""

import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
HISTORY_PATH = os.environ.get("TASK1_VIT_HISTORY_PATH", r"D:\FYP\runs_vit\train_vit_b16_best\history.json")
OUTPUT_DIR = os.environ.get("TASK1_OUTPUT_DIR", r"D:\FYP\Classifier\paper_results\task1_source_classification\visualizations")

def plot_training_curves(history_path, output_dir):
    """重新绘制训练曲线"""
    
    # 读取历史数据
    print(f"📂 读取训练历史: {history_path}")
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 1. Train Loss vs Val Loss (在同一张图上)
    print("📈 绘制 Train Loss vs Val Loss...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_validation_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {output_dir / 'training_validation_loss.png'}")
    
    # 2. Validation Accuracy
    print("📈 绘制 Validation Accuracy...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {output_dir / 'validation_accuracy.png'}")
    
    # 3. Validation F1 Score
    print("📈 绘制 Validation F1...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history['val_f1'], 'm-', label='Val F1', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {output_dir / 'validation_f1.png'}")
    
    # 4. Learning Rate
    print("📈 绘制 Learning Rate...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, history['lr'], 'orange', label='Learning Rate', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存: {output_dir / 'learning_rate_schedule.png'}")
    
    print("\n" + "="*60)
    print("✅ 所有训练曲线已重新生成!")
    print("="*60)
    print(f"📁 输出目录: {output_dir}")
    print("  - training_validation_loss.png")
    print("  - validation_accuracy.png")
    print("  - validation_f1.png")
    print("  - learning_rate_schedule.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    plot_training_curves(HISTORY_PATH, OUTPUT_DIR)
