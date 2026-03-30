#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-10-17 16:17
# @Author : xxx

import torch
import torch.nn as nn

from repgene.model.modelv17.repgene_components import (
    ResidualBlock,
    Adapter,
    ResidualFusionModule,
    AttentionFusionModule,
    TransformerFusionModule,
    MoEFusionModule
)

class RepGeneV17(nn.Module):
    """
    增强版多模态融合模型，包含重构解码器和分类头

    支持的编码器类型: ['mlp', 'residual', 'transformer', 'deep']
    支持的融合器类型: ['residual', 'attention', 'transformer', 'moe']
    支持的解码器类型: ['mlp', 'residual', 'transformer', 'deep']
    """

    def __init__(self, input_dims, adapter_type='mlp', encoder_type='transformer', fusion_type='attention',
                 decoder_type='transformer', dropout_rate=0.3, n_clusters_per_modality=None, 
                 encoder_layers=4, fuser_layers=4, decoder_layers=4):
        """
        参数:
            input_dims: 各模态输入维度字典
            encoder_type: 编码器类型
            fusion_type: 融合器类型
            decoder_type: 解码器类型
            dropout_rate: dropout率
            n_clusters_per_modality: 每个模态的聚类类别数量字典
            encoder_layers: 编码器transformer层数
            fuser_layers: 融合器transformer层数
            decoder_layers: 解码器transformer层数
        """
        super().__init__()
        self.modalities = list(input_dims.keys())
        self.num_modalities = len(self.modalities)
        self.output_dim = 256
        self.dropout_rate = dropout_rate
        self.adapter_type = adapter_type
        self.n_clusters_per_modality = n_clusters_per_modality
        self.decoder_type = decoder_type

        # 输入适配器
        self.InputAdapters = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.InputAdapters[mod] = Adapter(dim, self.output_dim, adapter_type=self.adapter_type,
                  num_layers=2, nhead=4, dropout_rate=0.3)
            # 共享编码器
        self.SharedEncoder = Encoder(
            self.output_dim, self.output_dim, encoder_type, dropout_rate, encoder_layers
        )

        # 融合模块
        self.Fuser = FusionModule(
            self.num_modalities, self.output_dim, fusion_type, dropout_rate, fuser_layers
        )
        self.fusion_dim = self.output_dim

        # 输出解码器
        self.OutputDecoders = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.OutputDecoders[mod] = Decoder(
                self.fusion_dim, dim, decoder_type, dropout_rate, decoder_layers
            )

        # 多模态分类头
        self.Classifiers = nn.ModuleDict()
        for mod in self.modalities:
            n_clusters = n_clusters_per_modality.get(mod, 1)
            self.Classifiers[mod] = ClassificationHead(
                self.fusion_dim, n_clusters, dropout_rate
            )

    def forward(self, inputs):
        # 处理存在的模态
        existing_mods = [mod for mod in self.modalities if mod in inputs]
        adjusted = {}
        encoded = {}

        # 各模态处理流程
        for mod in existing_mods:
            # 输入适配器
            x = self.InputAdapters[mod](inputs[mod])
            adjusted[mod] = x

            # 共享编码器
            enc = self.SharedEncoder(x)
            encoded[mod] = enc

        # 为缺失模态创建零填充
        batch_size = next(iter(inputs.values())).size(0)
        device = next(self.parameters()).device

        # 构建融合模块输入
        fusion_inputs = []
        for mod in self.modalities:
            if mod in existing_mods:
                fusion_inputs.append(encoded[mod])
            else:
                fusion_inputs.append(torch.zeros(batch_size, self.output_dim).to(device))

        # 融合计算
        fused_emb = self.Fuser(fusion_inputs)

        # 重构各模态嵌入
        reconstructed = {}
        for mod in self.modalities:
            reconstructed[mod] = self.OutputDecoders[mod](fused_emb)

        # 多模态分类预测
        cluster_preds = {}
        for mod in self.modalities:
            cluster_preds[mod] = self.Classifiers[mod](fused_emb)

        return {
            'adjusted': adjusted,
            'encoded': encoded,
            'fused': fused_emb,
            'reconstructed': reconstructed,
            'cluster_preds': cluster_preds
        }


class Encoder(nn.Module):
    """
    共享编码器模块

    支持的类型:
    - 'mlp': 多层感知机编码器
    - 'residual': 残差块编码器
    - 'transformer': Transformer编码器
    - 'deep': 深层残差编码器
    - 'lp': 线性投影编码器
    """

    def __init__(self, input_dim=512, output_dim=512, encoder_type='mlp',
                 dropout_rate=0.3, num_layers=4):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'residual':
            self.net = nn.Sequential(
                *[ResidualBlock(input_dim, dropout_rate) for _ in range(num_layers)]
            )
        elif encoder_type == 'deep':
            self.net = nn.Sequential(
                *[ResidualBlock(input_dim, dropout_rate) for _ in range(num_layers)]
            )
        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=4, dim_feedforward=512,
                dropout=dropout_rate, batch_first=True
            )
            self.net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif encoder_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, output_dim)
            )
        else:
            raise ValueError(f"未知编码器类型: {encoder_type}")

    def forward(self, x):
        if self.encoder_type == 'transformer':
            return self.net(x.unsqueeze(1)).squeeze(1)
        return self.net(x)

class FusionModule(nn.Module):
    """
    融合模块

    支持的类型:
    - 'residual': 残差融合
    - 'attention': 注意力融合
    - 'transformer': Transformer融合
    - 'moe': 专家混合融合
    """

    def __init__(self, num_modalities, input_dim=512, fusion_type='residual',
                 dropout_rate=0.3, num_layers=4, **kwargs):
        super().__init__()
        self.num_modalities = num_modalities
        self.input_dim = input_dim

        if fusion_type == 'attention':
            self.net = AttentionFusionModule(
                num_modalities, input_dim, dropout_rate=dropout_rate, num_layers=num_layers
            )
        elif fusion_type == 'transformer':
            nhead = kwargs.get('nhead', 8)
            dim_feedforward = kwargs.get('dim_feedforward', input_dim*4)
            self.net = TransformerFusionModule(
                num_modalities, input_dim, num_layers=num_layers, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout_rate=dropout_rate
            )
        elif fusion_type == 'moe':
            num_experts = kwargs.get('num_experts', 4)
            expert_dim = kwargs.get('expert_dim', 256)
            self.net = MoEFusionModule(
                num_modalities, input_dim, num_experts=num_experts,
                expert_dim=expert_dim, dropout_rate=dropout_rate, num_layers=num_layers
            )
        else:  # 默认使用残差融合
            self.net = ResidualFusionModule(
                num_modalities, input_dim, dropout_rate=dropout_rate, num_layers=num_layers
            )

    def forward(self, inputs):
        return self.net(inputs)

class Decoder(nn.Module):
    """
    输出解码器模块

    支持的类型:
    - 'mlp': 多层感知机解码器
    - 'residual': 残差块解码器
    - 'transformer': Transformer解码器
    - 'deep': 深层解码器
    """

    def __init__(self, input_dim, output_dim, decoder_type='mlp',
                 dropout_rate=0.3, num_layers=4):
        super().__init__()
        self.decoder_type = decoder_type

        if decoder_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, output_dim)
            )
        elif decoder_type == 'residual':
            self.net = nn.Sequential(
                *[ResidualBlock(input_dim, dropout_rate) for _ in range(num_layers)],
                nn.Linear(input_dim, output_dim)
            )
        elif decoder_type == 'transformer':
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=512,
                dropout=dropout_rate,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.query_embed = nn.Parameter(torch.randn(1, input_dim))
            self.output_proj = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, output_dim)
            )
        elif decoder_type == 'deep':
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, output_dim)
            )
        else:
            raise ValueError(f"未知解码器类型: {decoder_type}")

    def forward(self, x):
        if self.decoder_type == 'transformer':
            batch_size = x.size(0)
            memory = x.unsqueeze(1)
            query = self.query_embed.repeat(batch_size, 1, 1)
            decoded = self.transformer_decoder(query, memory)
            output = self.output_proj(decoded.squeeze(1))
            return output
        else:
            return self.net(x)


class ClassificationHead(nn.Module):
    """分类头模块"""

    def __init__(self, input_dim, n_classes, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)
