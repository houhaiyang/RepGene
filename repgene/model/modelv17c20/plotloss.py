
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages


def plot_training_history(history_path):
    # 加载训练历史
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # 获取实际的epoch数量
    num_epochs = len(history['train_total_loss'])
    epochs = range(1, num_epochs + 1)

    # 创建PDF文件
    pdf_path = os.path.join(os.path.dirname(history_path), 'training_curves.pdf')
    with PdfPages(pdf_path) as pdf:
        # 1. 总损失图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_total_loss'], 'b-', label='Train Total Loss')
        plt.plot(epochs, history['val_total_loss'], 'r-', label='Validation Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 3. 重构损失图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_reconstruction_loss'], 'b-', label='Train Reconstruction Loss')
        plt.plot(epochs, history['val_reconstruction_loss'], 'r-', label='Validation Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 4. 分类损失图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_classification_loss'], 'b-', label='Train Classification Loss')
        plt.plot(epochs, history['val_classification_loss'], 'r-', label='Validation Classification Loss')
        plt.title('Classification Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 5. 学习率图（对数坐标）
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['lr'], color='purple', linestyle='-', label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate (log scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    print(f"训练曲线已保存到: {pdf_path}")


if __name__ == '__main__':
    # 使用示例
    history_path = 'models/Homo_sapiens/2025-10-17-M5-V17/training_history.pkl'  # 替换为你的实际路径

    if os.path.exists(history_path):
        plot_training_history(history_path)
    else:
        print(f"错误: 文件 {history_path} 不存在")
