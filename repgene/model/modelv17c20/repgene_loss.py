#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-10-17 16:17
# @Author : Haiyang HOU

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_contrast=1.0, lambda_rec=1.0,
                 lambda_cls=1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_contrast = lambda_contrast
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)  # 添加ignore_index处理缺失标签
        self.mse_loss = nn.MSELoss()

    def contrastive_loss(self, encoded_dict):
        """模态间对比损失"""
        total_loss = 0
        pair_count = 0

        encoded_list = list(encoded_dict.values())

        for i in range(len(encoded_list)):
            for j in range(i + 1, len(encoded_list)):
                # 计算模态i和j的对比损失
                A_norm = F.normalize(encoded_list[i], dim=1)
                B_norm = F.normalize(encoded_list[j], dim=1)
                logits = torch.mm(A_norm, B_norm.t()) / self.temperature

                labels = torch.arange(A_norm.size(0)).to(A_norm.device)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss
                pair_count += 1

        return total_loss / max(1, pair_count)

    def reconstruction_loss(self, reconstructed_dict, original_dict):
        """重构损失"""
        total_loss = 0
        count = 0

        for mod in reconstructed_dict:
            if mod in original_dict:
                loss = self.mse_loss(reconstructed_dict[mod], original_dict[mod])
                total_loss += loss
                count += 1

        return total_loss / max(1, count)

    def classification_loss(self, pred_dict, target_dict):
        """修改分类损失以处理多模态分类"""
        total_loss = 0
        count = 0

        for mod in pred_dict:
            if mod in target_dict:
                pred = pred_dict[mod]
                target = target_dict[mod]

                # 只计算有效标签的损失 (target != -1)
                valid_mask = target != -1
                if valid_mask.any():
                    valid_pred = pred[valid_mask]
                    valid_target = target[valid_mask]
                    loss = self.ce_loss(valid_pred, valid_target)
                    total_loss += loss
                    count += 1

        return total_loss / max(1, count)  # 平均所有有效模态的损失

    def forward(self, outputs, inputs, cluster_labels):
        """
        修改以处理多模态分类任务
        """
        # 计算各项损失
        contrast_loss = self.contrastive_loss(outputs['encoded'])
        rec_loss = self.reconstruction_loss(outputs['reconstructed'], inputs)
        cls_loss = self.classification_loss(outputs['cluster_preds'], cluster_labels)  # 修改为cluster_preds

        # 加权求和
        total_loss = (self.lambda_contrast * contrast_loss +
                      self.lambda_rec * rec_loss +
                      self.lambda_cls * cls_loss)

        return {
            'total': total_loss,
            'contrast': contrast_loss,
            'reconstruction': rec_loss,
            'classification': cls_loss
        }
