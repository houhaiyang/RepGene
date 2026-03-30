#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-10-17 16:17
# @Author : xxx

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块基础组件"""
    def __init__(self, features, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(features, features),
            nn.BatchNorm1d(features)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


def Adapter(input_dim, output_dim=512, adapter_type='mlp', num_layers=2, nhead=4, dropout_rate=0.3):
    """创建输入适配器，支持多种类型

    参数:
        input_dim: 输入维度
        output_dim: 输出维度，默认512
        adapter_type: 适配器类型，可选['mlp', 'transformer', 'residual']
        num_layers: Transformer层数（仅transformer类型有效）
        nhead: Transformer头数（仅transformer类型有效）
        dropout_rate: dropout率
    """
    if adapter_type == 'transformer':
        return TransformerAdapter(input_dim, output_dim, num_layers, nhead, dropout_rate)
    elif adapter_type == 'residual':
        return ResidualAdapter(input_dim, output_dim, dropout_rate)
    else:  # 默认mlp适配器
        return MLPAdapter(input_dim, output_dim, dropout_rate)


class MLPAdapter(nn.Module):
    """MLP适配器"""

    def __init__(self, input_dim, output_dim=512, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ResidualAdapter(nn.Module):
    """残差适配器"""

    def __init__(self, input_dim, output_dim=512, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            *[ResidualBlock(output_dim, dropout_rate) for _ in range(2)],
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerAdapter(nn.Module):
    """Transformer适配器"""

    def __init__(self, input_dim, output_dim=512, num_layers=2, nhead=4, dropout_rate=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=output_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # 输入投影和批归一化
        x = self.input_proj(x)
        x = self.bn(x)

        # 添加序列维度并应用Transformer
        # 形状: (batch_size, 1, output_dim)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        # 输出投影
        x = self.output_proj(x)
        return x


class ResidualFusionModule(nn.Module):
    """残差融合模块"""
    def __init__(self, num_modalities, input_dim=512, dropout_rate=0.3, num_layers=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.input_dim = input_dim

        # 模态存在性编码器
        self.presence_encoder = nn.Sequential(
            nn.Linear(num_modalities, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # 残差融合网络
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(input_dim, dropout_rate) for _ in range(num_layers)]
        )

        # 输出投影层
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        batch_size = inputs[0].size(0)
        device = inputs[0].device

        # 计算模态存在性掩码
        presence_mask = []
        for i, mod_input in enumerate(inputs):
            mod_present = torch.any(mod_input != 0, dim=1).float().unsqueeze(1)
            presence_mask.append(mod_present)

        presence_mask = torch.cat(presence_mask, dim=1)

        # 编码模态存在性信息
        presence_encoding = self.presence_encoder(presence_mask)

        # 计算加权平均融合
        weighted_sum = torch.zeros(batch_size, self.input_dim).to(device)
        total_weight = 1e-6

        for i, mod_input in enumerate(inputs):
            mod_weight = presence_mask[:, i].unsqueeze(1) * torch.norm(mod_input, dim=1, keepdim=True)
            weighted_sum += mod_weight * mod_input
            total_weight += mod_weight

        # 加权平均
        fused = weighted_sum / total_weight
        fused = fused + presence_encoding

        # 通过残差网络
        fused = self.residual_blocks(fused)
        fused = self.output_proj(fused)

        return fused


class AttentionFusionModule(nn.Module):
    """注意力融合模块"""
    def __init__(self, num_modalities, input_dim=512, hidden_dim=256,
                 dropout_rate=0.3, num_layers=2):
        super().__init__()
        self.num_modalities = num_modalities
        self.input_dim = input_dim

        # 模态存在性编码
        self.presence_encoder = nn.Sequential(
            nn.Linear(num_modalities, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, hidden_dim)
        )

        # 内容注意力机制
        self.content_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=8, dropout=dropout_rate, batch_first=True
        )

        # 模态间关系建模
        self.cross_modal_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, dropout=dropout_rate,
                                batch_first=True) for _ in range(num_modalities)
        ])

        # 门控融合机制
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 残差细化
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(input_dim, dropout_rate) for _ in range(num_layers)]
        )

    def forward(self, inputs):
        batch_size = inputs[0].size(0)
        device = inputs[0].device

        # 模态存在性检测
        presence_mask = []
        valid_modalities = []
        for i, mod_input in enumerate(inputs):
            mod_present = torch.any(mod_input != 0, dim=1).float().unsqueeze(1)
            presence_mask.append(mod_present)
            if torch.any(mod_present > 0):
                valid_modalities.append((i, mod_input))

        presence_mask = torch.cat(presence_mask, dim=1)

        # 内容感知注意力
        if len(valid_modalities) > 0:
            valid_inputs = torch.stack([mod_input for _, mod_input in valid_modalities], dim=1)
            attended, _ = self.content_attention(valid_inputs, valid_inputs, valid_inputs)

            cross_modal_features = []
            for idx, (mod_idx, original_input) in enumerate(valid_modalities):
                query = original_input.unsqueeze(1)
                key_value = attended
                cross_attended, _ = self.cross_modal_attention[mod_idx](query, key_value, key_value)
                cross_modal_features.append(cross_attended.squeeze(1))

            # 加权融合
            weights = []
            for feature in cross_modal_features:
                quality_score = torch.norm(feature, dim=1, keepdim=True)
                presence_weight = presence_mask[:, mod_idx].unsqueeze(1)
                weight = quality_score * presence_weight
                weights.append(weight)

            weights = torch.softmax(torch.cat(weights, dim=1), dim=1)
            fused = sum(w.unsqueeze(2) * f.unsqueeze(1) for w, f in
                        zip(weights.unbind(dim=1), cross_modal_features))
            fused = fused.sum(dim=1)
        else:
            fused = torch.zeros(batch_size, self.input_dim).to(device)

        # 门控残差连接
        presence_encoding = self.presence_encoder(presence_mask)
        gate = self.gate_network(torch.cat([fused, presence_encoding], dim=1))
        fused = gate * fused + (1 - gate) * presence_encoding

        # 残差细化
        fused = self.residual_blocks(fused)

        return fused


class TransformerFusionModule(nn.Module):
    """Transformer融合模块"""
    def __init__(self, num_modalities, input_dim=512, num_layers=4, nhead=4,
                 dim_feedforward=512, dropout_rate=0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.input_dim = input_dim

        # 模态类型编码
        self.modal_embeddings = nn.Embedding(num_modalities, input_dim)

        # 存在性编码
        self.presence_encoder = nn.Linear(num_modalities, input_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 融合头
        self.fusion_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, inputs):
        batch_size = inputs[0].size(0)
        device = inputs[0].device

        # 构建Transformer输入序列
        sequence = []
        presence_mask = []

        for mod_idx, mod_input in enumerate(inputs):
            modal_emb = self.modal_embeddings(torch.tensor(mod_idx).to(device))
            modal_emb = modal_emb.expand(batch_size, -1)

            mod_present = torch.any(mod_input != 0, dim=1).float().unsqueeze(1)
            presence_mask.append(mod_present)

            if torch.any(mod_present > 0):
                token = mod_input + modal_emb
            else:
                token = modal_emb

            sequence.append(token.unsqueeze(1))

        # 添加存在性编码
        presence_mask_tensor = torch.cat(presence_mask, dim=1)
        presence_token = self.presence_encoder(presence_mask_tensor).unsqueeze(1)
        sequence.append(presence_token)

        # 堆叠序列
        sequence = torch.cat(sequence, dim=1)

        # Transformer编码
        encoded = self.transformer_encoder(sequence)

        # 使用存在性token作为融合表示
        fused = self.fusion_head(encoded[:, -1, :])

        return fused


class MoEFusionModule(nn.Module):
    """专家混合融合模块"""
    def __init__(self, num_modalities, input_dim=512, num_experts=4,
                 expert_dim=256, dropout_rate=0.3, num_layers=2):
        super().__init__()
        self.num_modalities = num_modalities
        self.input_dim = input_dim
        self.num_experts = num_experts

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(num_modalities, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

        # 残差连接
        self.residual = nn.Sequential(
            *[ResidualBlock(input_dim, dropout_rate) for _ in range(num_layers)]
        )

    def forward(self, inputs):
        batch_size = inputs[0].size(0)
        device = inputs[0].device

        # 计算模态存在性
        presence_mask = []
        weighted_inputs = []

        for i, mod_input in enumerate(inputs):
            mod_present = torch.any(mod_input != 0, dim=1).float().unsqueeze(1)
            presence_mask.append(mod_present)

            weight = mod_present * torch.norm(mod_input, dim=1, keepdim=True)
            weighted_inputs.append(weight * mod_input)

        presence_mask = torch.cat(presence_mask, dim=1)

        # 基础融合
        base_fused = sum(weighted_inputs) / (sum([w for w, _ in weighted_inputs]) + 1e-6)

        # 门控网络计算专家权重
        gate_weights = self.gate_network(presence_mask)

        # 专家混合
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(base_fused)
            expert_outputs.append(expert_out.unsqueeze(1))

        expert_outputs = torch.cat(expert_outputs, dim=1)

        # 加权组合
        fused = torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=1)

        # 残差连接
        fused = self.residual(fused)

        return fused

