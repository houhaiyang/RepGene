#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-10-17
# @Author : Haiyang HOU

import os
import torch.optim as optim
import json
import torch
import torch.nn as nn
import math
import pickle

# 导入自定义模块
from repgene.model.modelv17c20.repgene_model import RepGeneV17
from repgene.model.modelv17c20.repgene_loss import MultiTaskLoss
from repgene.dataPreprocessing.read_input_embeddings__Homo_sapiens_M5_ClusterV17C20 import load_dataloader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def main():
    # 配置参数
    config = {
        'modalities': ['DNA', 'RNA', 'protein', 'text', 'singlecell'],
        'data_dir': 'data/dataloader/Homo_sapiens-M5-ModalityClusters',
        'model_dir': 'models/Homo_sapiens/2025-10-17-M5-V17',

        # 模型架构配置
        'adapter_type': 'mlp',
        'encoder_type': 'transformer',
        'fusion_type': 'residual',
        'decoder_type': 'transformer',
        'dropout_rate': 0.3,

        # Transformer层数配置
        'encoder_layers': 2,
        'fuser_layers': 4,
        'decoder_layers': 2,

        # 输入维度配置
        'input_dims': {
            'DNA': 4096,
            'RNA': 2560,
            'protein': 1152,
            'text': 1024,
            'singlecell': 512
        },

        'n_clusters_per_modality': {
            'DNA': 5,
            'RNA': 5,
            'protein': 5,
            'text': 5,
            'singlecell': 5
        },

        # 损失函数配置 (取消分阶段训练)
        'temperature': 0.07,
        'lambda_contrast': 1.0,  # 新增参数
        'lambda_rec': 0.1,
        'lambda_cls': 2.0,

        # 训练参数
        'batch_size': 512,
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'num_epochs': 100,

        # 学习率调度配置
        'warmup_steps': 500,
        'stable_steps': 3000,
        'min_lr': 1e-6,

        # 训练监控配置
        'early_stop_patience': 5,
        'log_interval': 103
    }

    # 创建保存目录
    os.makedirs(config['model_dir'], exist_ok=True)

    # 保存配置
    config_path = os.path.join(config['model_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"保存配置: {config_path}")

    # 加载数据
    print("加载数据...")
    train_dataloader = load_dataloader(os.path.join(config['data_dir'], 'train'))
    val_dataloader = load_dataloader(os.path.join(config['data_dir'], 'val'))

    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    # 创建模型
    print("\n创建增强融合模型(v17)...")
    # 创建模型和损失函数
    model = RepGeneV17(
        input_dims=config['input_dims'],
        adapter_type=config['adapter_type'],
        encoder_type=config['encoder_type'],
        fusion_type=config['fusion_type'],
        decoder_type=config['decoder_type'],
        dropout_rate=config['dropout_rate'],
        n_clusters_per_modality=config['n_clusters_per_modality'],  # 修改参数名
        encoder_layers=config['encoder_layers'],
        fuser_layers=config['fuser_layers'],  # 修改参数名
        decoder_layers=config['decoder_layers']
    )

    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"编码器类型: {config['encoder_type']}, 层数: {config['encoder_layers']}")
    print(f"融合器类型: {config['fusion_type']}, 层数: {config['fuser_layers']}")
    print(f"解码器类型: {config['decoder_type']}, 层数: {config['decoder_layers']}")

    # 创建损失函数 (取消分阶段训练)
    criterion = MultiTaskLoss(
        temperature=config['temperature'],
        lambda_contrast=config['lambda_contrast'],
        lambda_rec=config['lambda_rec'],
        lambda_cls=config['lambda_cls']
    )

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 训练模型 (简化示例)
    print("\n开始训练...")
    # 这里需要根据实际的数据加载器进行调整
    history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=config['num_epochs'],
        save_dir=config['model_dir'],
        config=config,  # 传递config用于模态列表
        warmup_steps=config['warmup_steps'],
        stable_steps=config['stable_steps'],
        min_lr=config['min_lr'],
        early_stop_patience=config['early_stop_patience'],
        log_interval=config['log_interval']
    )

    # 保存训练历史
    save_training_history(history, os.path.join(config['model_dir'], 'training_history.pkl'))

    # 测试模型
    print("\n测试最佳模型...")
    test_dataloader = load_dataloader(os.path.join(config['data_dir'], 'test'))

    # 加载最佳模型
    best_model_path = os.path.join(config['model_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)

        # 创建新模型实例
        test_model = RepGeneV17(
            input_dims=config['input_dims'],
            adapter_type=config['adapter_type'],
            encoder_type=config['encoder_type'],
            fusion_type=config['fusion_type'],
            decoder_type=config['decoder_type'],
            dropout_rate=config['dropout_rate'],
            n_clusters_per_modality=config['n_clusters_per_modality'],
            encoder_layers=config['encoder_layers'],
            fuser_layers=config['fuser_layers'],
            decoder_layers=config['decoder_layers']
        ).to(device)

        # 加载状态字典
        test_model.load_state_dict(checkpoint['model_state_dict'])
        test_model.eval()

        # 在测试集上评估
        test_losses = validate_model(
            test_model,
            test_dataloader,
            criterion,
            config
        )

        print("\n测试结果:")
        print(f"总损失: {test_losses['total']:.4f}")
        print(f"对比损失: {test_losses['contrast']:.4f}")
        print(f"重构损失: {test_losses['reconstruction']:.4f}")
        print(f"分类损失: {test_losses['classification']:.4f}")

        # 保存测试结果
        test_results = {
            'losses': test_losses,
            'config': config,
            'epoch': checkpoint['epoch']
        }
        test_path = os.path.join(config['model_dir'], 'test_results.json')
        with open(test_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"保存测试结果: {test_path}")
    else:
        print(f"未找到最佳模型: {best_model_path}")

    print("\n训练完成!")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=100,
                save_dir='models', config=None, warmup_steps=300, stable_steps=1000,
                min_lr=1e-6, early_stop_patience=10, log_interval=100):
    """训练函数实现"""
    os.makedirs(save_dir, exist_ok=True)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    model.to(device)

    # 初始学习率
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"初始学习率: {initial_lr:.1e}, 最小学习率: {min_lr:.1e}")

    # 训练统计
    history = {
        'train_total_loss': [],
        'train_contrast_loss': [],
        'train_reconstruction_loss': [],
        'train_classification_loss': [],
        'val_total_loss': [],
        'val_contrast_loss': [],
        'val_reconstruction_loss': [],
        'val_classification_loss': [],
        'lr': []
    }

    best_val_loss = float('inf')
    no_improve_count = 0

    # 计算总训练步数
    total_batches_per_epoch = len(dataloaders['train'])
    total_train_steps = num_epochs * total_batches_per_epoch
    print(f"预计总训练步数: {total_train_steps}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 60)

        # 训练阶段
        model.train()
        epoch_train_total = 0.0
        epoch_train_contrast = 0.0
        epoch_train_reconstruction = 0.0
        epoch_train_classification = 0.0

        for batch_idx, batch in enumerate(dataloaders['train']):
            total_steps = epoch * total_batches_per_epoch + batch_idx

            # 学习率调度
            if total_steps < warmup_steps:
                lr_scale = min(1.0, total_steps / warmup_steps)
                current_lr = initial_lr * lr_scale
            elif total_steps < warmup_steps + stable_steps:
                current_lr = initial_lr
            else:
                decay_progress = (total_steps - warmup_steps - stable_steps) / \
                                 (total_train_steps - warmup_steps - stable_steps)
                decay_progress = min(1.0, decay_progress)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                current_lr = min_lr + (initial_lr - min_lr) * cosine_decay

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 修改输入数据准备
            batch_mods = {mod: tensor.to(device) for mod, tensor in batch.items()
                          if mod not in ['genes', 'cluster']}

            # 修改聚类标签处理 - 现在每个模态都有独立的标签
            cluster_labels = {}
            for mod in config['modalities']:
                if mod in batch['cluster']:
                    cluster_labels[mod] = batch['cluster'][mod].to(device)
                else:
                    # 处理缺失模态的情况
                    cluster_labels[mod] = torch.full((batch_mods[list(batch_mods.keys())[0]].size(0),), -1).to(device)

            # 前向传播
            outputs = model(batch_mods)
            losses = criterion(outputs, batch_mods, cluster_labels)  # 传入字典形式的标签

            # 反向传播
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            # 更新统计
            epoch_train_total += losses['total'].item()
            epoch_train_contrast += losses['contrast'].item()
            epoch_train_reconstruction += losses['reconstruction'].item()
            epoch_train_classification += losses['classification'].item()

            # 定期打印训练信息
            if batch_idx % log_interval == 0:
                avg_total = epoch_train_total / (batch_idx + 1)
                avg_contrast = epoch_train_contrast / (batch_idx + 1)
                avg_reconstruction = epoch_train_reconstruction / (batch_idx + 1)
                avg_classification = epoch_train_classification / (batch_idx + 1)

                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx}/{total_batches_per_epoch} | "
                    f"Loss: {avg_total:.4f} | "
                    f"Ctr: {avg_contrast:.4f} | Rec: {avg_reconstruction:.4f} | "
                    f"Cls: {avg_classification:.4f} | LR: {current_lr:.3e}"
                )

        # 计算epoch平均训练损失
        avg_epoch_total = epoch_train_total / total_batches_per_epoch
        avg_epoch_contrast = epoch_train_contrast / total_batches_per_epoch
        avg_epoch_reconstruction = epoch_train_reconstruction / total_batches_per_epoch
        avg_epoch_classification = epoch_train_classification / total_batches_per_epoch

        history['train_total_loss'].append(avg_epoch_total)
        history['train_contrast_loss'].append(avg_epoch_contrast)
        history['train_reconstruction_loss'].append(avg_epoch_reconstruction)
        history['train_classification_loss'].append(avg_epoch_classification)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_epoch_total:.4f} | "
              f"Ctr: {avg_epoch_contrast:.4f} | Rec: {avg_epoch_reconstruction:.4f} | "
              f"Cls: {avg_classification:.4f}")

        # 验证阶段
        val_losses = validate_model(model, dataloaders['val'], criterion, config)
        val_total = val_losses['total']
        val_contrast = val_losses['contrast']
        val_reconstruction = val_losses['reconstruction']
        val_classification = val_losses['classification']

        history['val_total_loss'].append(val_total)
        history['val_contrast_loss'].append(val_contrast)
        history['val_reconstruction_loss'].append(val_reconstruction)
        history['val_classification_loss'].append(val_classification)

        print(f"Epoch {epoch + 1} | Val Loss: {val_total:.4f} | "
              f"Ctr: {val_contrast:.4f} | Rec: {val_reconstruction:.4f} | "
              f"Cls: {val_classification:.4f}")

        # 检查是否为最佳模型
        if val_total < best_val_loss:
            best_val_loss = val_total
            no_improve_count = 0

            # 保存最佳模型
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total,
                'config': config
            }, save_path)
            print(f"保存最佳模型 (Epoch {epoch + 1}, Val Loss: {val_total:.4f})")
        else:
            no_improve_count += 1
            print(f"验证损失未改善 ({no_improve_count}/{early_stop_patience})")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'model_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total,
                'config': config
            }, save_path)
            print(f"保存第 {epoch + 1} 个Epoch的模型")

        # 早停检查
        if no_improve_count >= early_stop_patience:
            print(f"早停触发! 连续 {early_stop_patience} 个Epoch验证损失未改善")
            break

    # 保存最终模型
    save_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_total,
        'config': config
    }, save_path)
    print(f"保存最终模型: {save_path}")

    return history


def validate_model(model, dataloader, criterion, config):
    """验证函数 - 同样修改聚类标签处理"""
    model.eval()
    total_loss = 0.0
    total_contrast = 0.0
    total_reconstruction = 0.0
    total_classification = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_mods = {mod: tensor.to(device) for mod, tensor in batch.items()
                          if mod not in ['genes', 'cluster']}

            # 修改聚类标签处理
            cluster_labels = {}
            for mod in config['modalities']:
                if mod in batch['cluster']:
                    cluster_labels[mod] = batch['cluster'][mod].to(device)
                else:
                    cluster_labels[mod] = torch.full((batch_mods[list(batch_mods.keys())[0]].size(0),), -1).to(device)

            batch_size = len(batch['genes'])
            outputs = model(batch_mods)
            losses = criterion(outputs, batch_mods, cluster_labels)

            total_loss += losses['total'].item() * batch_size
            total_contrast += losses['contrast'].item() * batch_size
            total_reconstruction += losses['reconstruction'].item() * batch_size
            total_classification += losses['classification'].item() * batch_size
            total_samples += batch_size

    # 计算平均值
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_contrast = total_contrast / total_samples if total_samples > 0 else 0.0
    avg_reconstruction = total_reconstruction / total_samples if total_samples > 0 else 0.0
    avg_classification = total_classification / total_samples if total_samples > 0 else 0.0

    return {
        'total': avg_loss,
        'contrast': avg_contrast,
        'reconstruction': avg_reconstruction,
        'classification': avg_classification
    }


def save_training_history(history, filepath):
    """保存训练历史"""
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)


def save_model(model, optimizer, epoch, val_loss, config, save_path):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config,  # 保存完整配置
        'n_clusters_per_modality': config['n_clusters_per_modality']  # 单独保存聚类配置
    }, save_path)


def load_model(model_path, device='cuda'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)

    # 使用保存的配置重建模型
    model = RepGeneV17(
        input_dims=checkpoint['config']['input_dims'],
        adapter_type=checkpoint['config']['adapter_type'],
        encoder_type=checkpoint['config']['encoder_type'],
        fusion_type=checkpoint['config']['fusion_type'],
        decoder_type=checkpoint['config']['decoder_type'],
        dropout_rate=checkpoint['config']['dropout_rate'],
        n_clusters_per_modality=checkpoint['n_clusters_per_modality'],  # 使用保存的聚类配置
        encoder_layers=checkpoint['config']['encoder_layers'],
        fuser_layers=checkpoint['config']['fuser_layers'],
        decoder_layers=checkpoint['config']['decoder_layers']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


if __name__ == '__main__':
    main()