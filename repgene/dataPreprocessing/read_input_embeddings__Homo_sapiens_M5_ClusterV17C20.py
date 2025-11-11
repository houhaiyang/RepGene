#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-10-17 16:17
# @Author : Haiyang HOU

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

# 设置随机种子保证可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def perform_modality_clustering(embeddings_vectors: Dict[str, pd.DataFrame],
                                n_clusters_per_modality: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    对每个模态的嵌入进行聚类

    参数:
        embeddings_vectors: 各模态的嵌入向量字典
        n_clusters_per_modality: 每个模态的聚类数量

    返回:
        字典: {modality: 聚类标签数组}
    """
    modality_clusters = {}

    for modality, n_clusters in n_clusters_per_modality.items():
        if modality not in embeddings_vectors:
            print(f"警告: 模态 {modality} 不在嵌入数据中，跳过聚类")
            continue

        # 获取嵌入数据
        embeddings = embeddings_vectors[modality].values

        # 标准化数据
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        clusters = kmeans.fit_predict(embeddings_scaled)

        modality_clusters[modality] = clusters
        print(f"模态 {modality} 聚类完成，共 {n_clusters} 个类别")

    return modality_clusters


def create_modality_cluster_mappings(embeddings_genes: Dict[str, List[str]],
                                     modality_clusters: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    """
    为每个模态创建基因到聚类标签的映射

    参数:
        embeddings_genes: 各模态的基因名称列表字典
        modality_clusters: 各模态的聚类标签字典

    返回:
        字典: {modality: {gene: cluster_label}}
    """
    modality_cluster_mappings = {}

    for modality, clusters in modality_clusters.items():
        if modality not in embeddings_genes:
            continue

        gene_to_cluster = {}
        for idx, gene in enumerate(embeddings_genes[modality]):
            gene_to_cluster[gene] = clusters[idx]

        modality_cluster_mappings[modality] = gene_to_cluster

        # 打印每个模态的聚类分布
        cluster_counts = Counter(gene_to_cluster.values())
        print(f"模态 {modality} 聚类分布:")
        for cluster_id in sorted(cluster_counts.keys()):
            print(f"  类别 {cluster_id}: {cluster_counts[cluster_id]} 个基因")

    return modality_cluster_mappings


# 加载嵌入数据和基因名称映射的函数保持不变
def load_embeddings_and_gene_mappings(base_path: str,
                                      embeddings_files: Dict[str, str],
                                      csv_files: Dict[str, str]) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    加载嵌入数据和基因名称映射

    参数:
        base_path: 基础路径
        embeddings_files: 各模态嵌入文件路径字典
        csv_files: 各模态基因名称映射文件路径字典

    返回:
        embeddings_data: 各模态嵌入数据字典
        gene_mappings: 各模态序列ID到基因名称的映射字典
    """
    embeddings_data = {}
    gene_mappings = {}

    for modality in embeddings_files:
        # 构建完整文件路径
        embed_path = os.path.join(base_path, embeddings_files[modality])

        # 读取嵌入数据
        embeddings_data[modality] = pd.read_csv(embed_path, index_col=0)

        # 处理基因名称映射
        if modality in ['text', 'singlecell']:
            # text和singlecell模态的行名就是基因名
            if modality == 'singlecell':
                # 单细胞基因名需要去除后缀
                gene_names = [name.split('-')[0] for name in embeddings_data[modality].index]
            else:
                gene_names = embeddings_data[modality].index.tolist()

            gene_mappings[modality] = gene_names
        else:
            # 其他模态需要从单独的csv文件中获取基因名
            csv_path = os.path.join(base_path, csv_files[modality])
            csv_data = pd.read_csv(csv_path)

            # 确保csv文件中有'gene'列
            if 'gene' not in csv_data.columns:
                raise ValueError(f"{modality}的CSV文件缺少'gene'列")

            # 获取序列ID到基因名的映射
            seq_ids = embeddings_data[modality].index
            gene_mapping = csv_data.set_index('id')['gene'].to_dict()

            # 确保所有序列ID都有对应的基因名
            gene_names = []
            for seq_id in seq_ids:
                if seq_id not in gene_mapping:
                    raise ValueError(f"{modality}模态中序列ID {seq_id} 在CSV文件中找不到对应基因名")
                gene_names.append(gene_mapping[seq_id])

            gene_mappings[modality] = gene_names

    return embeddings_data, gene_mappings


class ResamplingMultiModalGeneDatasetNotAll2(Dataset):
    def __init__(self, embeddings_vectors: Dict[str, pd.DataFrame],
                 embeddings_genes: Dict[str, List[str]],
                 samples: Optional[List[Dict[str, Tuple[str, int]]]] = None,
                 modality_cluster_mappings: Optional[Dict[str, Dict[str, int]]] = None):
        """
        新的多模态基因数据集，确保每个基因的每个模态序列至少使用一次
        修改：将五个模态的聚类标签整合到一个cluster字段中，cluster字段是一个字典

        参数:
            embeddings_vectors: 各模态的嵌入向量字典
            embeddings_genes: 各模态的基因名称列表字典
            samples: 可选的样本列表，如果为None则创建优化样本
            modality_cluster_mappings: 各模态的基因到聚类标签的映射字典
        """
        self.modalities = list(embeddings_vectors.keys())
        self.embeddings = embeddings_vectors
        self.embeddings_genes = embeddings_genes
        self.modality_cluster_mappings = modality_cluster_mappings or {}

        # 构建基因到各模态索引的完整映射
        self.gene_to_indices = self._build_complete_gene_index_mapping(embeddings_genes)

        # 创建优化样本列表
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._create_optimized_samples()

    def _build_complete_gene_index_mapping(self, embeddings_genes: Dict[str, List[str]]) -> Dict[
        str, Dict[str, List[int]]]:
        """
        构建完整的基因索引映射
        """
        gene_dict = defaultdict(lambda: defaultdict(list))

        for mod in self.modalities:
            for idx, gene in enumerate(embeddings_genes[mod]):
                gene_dict[gene][mod].append(idx)

        return dict(gene_dict)

    def _create_optimized_samples(self) -> List[Dict[str, Tuple[str, int]]]:
        """
        创建优化样本，确保每个基因的每个模态序列至少使用一次
        每个样本包含一个基因的所有模态序列
        """
        samples = []

        for gene, mod_indices in self.gene_to_indices.items():
            # 只处理在所有模态都有至少一个序列的基因
            if not all(mod in mod_indices and mod_indices[mod] for mod in self.modalities):
                continue

            # 确定每个模态需要采样的序列数（取各模态序列数的最大值）
            max_count = max(len(mod_indices[mod]) for mod in self.modalities)

            # 为每个模态创建扩展序列列表
            expanded_indices = {}
            for mod in self.modalities:
                indices = mod_indices[mod]
                # 如果序列数不足max_count，则循环扩展
                if len(indices) < max_count:
                    expanded = indices * (max_count // len(indices))
                    expanded += indices[:max_count % len(indices)]
                else:
                    expanded = indices.copy()
                # 随机打乱扩展后的序列
                random.shuffle(expanded)
                expanded_indices[mod] = expanded

            # 创建样本
            for i in range(max_count):
                sample = {'gene': gene}
                for mod in self.modalities:
                    sample[mod] = (gene, expanded_indices[mod][i])
                samples.append(sample)

        print(f"优化样本创建完成: 共 {len(samples)} 个样本")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取样本，将五个模态的聚类标签整合到cluster字段的字典中"""
        sample_info = self.samples[idx]
        gene = sample_info['gene']

        # 创建样本字典
        sample = {'gene': gene}

        # 创建cluster字典，包含五个模态的聚类标签
        cluster_dict = {}
        for mod in self.modalities:
            if mod in self.modality_cluster_mappings and gene in self.modality_cluster_mappings[mod]:
                cluster_dict[mod] = self.modality_cluster_mappings[mod][gene]
            else:
                cluster_dict[mod] = -1  # 无聚类标签

        # 将cluster字典存储到样本中
        sample['cluster'] = cluster_dict

        # 获取各模态的嵌入向量
        for mod in self.modalities:
            gene_name, embed_idx = sample_info[mod]
            embedding = self.embeddings[mod].iloc[embed_idx].values
            sample[mod] = torch.FloatTensor(embedding)

        return sample


def resampling_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    重采样模式的批处理函数

    返回:
        字典结构: {
            'genes': [gene1, gene2,...],  # 基因名称列表
            'DNA': tensor_shape(batch_size, dna_embed_dim),
            'RNA': tensor_shape(batch_size, rna_embed_dim),
            ... # 其他模态
            'cluster': {
                'DNA': tensor_shape(batch_size),
                'RNA': tensor_shape(batch_size),
                'protein': tensor_shape(batch_size),
                'text': tensor_shape(batch_size),
                'singlecell': tensor_shape(batch_size)
            }
        }
    """
    collated = {'genes': [item['gene'] for item in batch]}

    # 获取第一个样本的所有键
    all_keys = batch[0].keys()

    # 分离模态键和cluster键
    modality_keys = [key for key in all_keys if key not in ['gene', 'cluster']]

    # 处理模态嵌入
    for mod in modality_keys:
        collated[mod] = torch.stack([item[mod] for item in batch])

    # 处理cluster标签 - 将字典列表转换为字典的张量
    cluster_dict = {}
    modalities = ['DNA', 'RNA', 'protein', 'text', 'singlecell']

    for mod in modalities:
        cluster_labels = []
        for item in batch:
            if 'cluster' in item and mod in item['cluster']:
                cluster_labels.append(item['cluster'][mod])
            else:
                cluster_labels.append(-1)  # 默认值

        cluster_dict[mod] = torch.LongTensor(cluster_labels)

    collated['cluster'] = cluster_dict

    return collated


def create_resampling_dataloader(embeddings_vectors: Dict[str, pd.DataFrame],
                                 embeddings_genes: Dict[str, List[str]],
                                 batch_size: int = 512,
                                 shuffle: bool = True,
                                 samples: List[Dict] = None,
                                 modality_cluster_mappings: Optional[Dict[str, Dict[str, int]]] = None) -> DataLoader:
    """
    创建支持重采样的多模态基因数据加载器

    参数:
        embeddings_vectors: 各模态的嵌入向量字典
        embeddings_genes: 各模态的基因名称列表字典
        batch_size: 批量大小
        shuffle: 是否打乱数据
        samples: 可选样本列表
        modality_cluster_mappings: 各模态的基因到聚类标签的映射字典

    返回:
        DataLoader实例
    """
    dataset = ResamplingMultiModalGeneDatasetNotAll2(
        embeddings_vectors,
        embeddings_genes,
        samples=samples,
        modality_cluster_mappings=modality_cluster_mappings
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=resampling_collate_fn,
        num_workers=0,  # 在Windows上暂时设置 num_workers=0禁用多进程
        pin_memory=True
    )
    return loader


def split_samples(samples: List[Dict], train_ratio: float = 0.7) -> Tuple[List, List]:
    """
    将样本列表划分为训练集和验证集

    参数:
        samples: 完整的样本列表
        train_ratio: 训练集比例

    返回:
        train_samples: 训练集样本列表
        val_samples: 验证集样本列表
    """
    # 随机打乱样本
    random.shuffle(samples)

    # 计算划分点
    split_idx = int(len(samples) * train_ratio)

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"数据集划分: 训练集 {len(train_samples)} 个样本, 验证集 {len(val_samples)} 个样本")
    return train_samples, val_samples


def save_dataloader_config(dataloader: DataLoader,
                           embeddings_vectors: Dict[str, pd.DataFrame],
                           embeddings_genes: Dict[str, List[str]],
                           save_dir: str,
                           samples: List[Dict] = None,
                           modality_cluster_mappings: Optional[Dict[str, Dict[str, int]]] = None):
    """
    保存重建数据加载器所需的配置，包括样本列表和聚类信息

    参数:
        dataloader: 要保存的数据加载器
        embeddings_vectors: 各模态的嵌入向量字典
        embeddings_genes: 各模态的基因名称列表字典
        save_dir: 保存目录路径
        samples: 样本列表
        modality_cluster_mappings: 各模态的基因到聚类标签的映射字典
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存数据加载器配置
    dl_config = {
        'batch_size': dataloader.batch_size,
        'shuffle': isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
    }

    with open(os.path.join(save_dir, 'dataloader_config.json'), 'w') as f:
        json.dump(dl_config, f)

    # 2. 保存嵌入向量
    for modality, df in embeddings_vectors.items():
        df.to_pickle(os.path.join(save_dir, f'{modality}_embeddings.pkl'))

    # 3. 保存基因映射
    with open(os.path.join(save_dir, 'gene_mappings.pkl'), 'wb') as f:
        pickle.dump(embeddings_genes, f)

    # 4. 保存样本列表（如果提供）
    if samples is not None:
        with open(os.path.join(save_dir, 'samples.pkl'), 'wb') as f:
            pickle.dump(samples, f)

    # 5. 保存聚类信息（如果提供）
    if modality_cluster_mappings is not None:
        with open(os.path.join(save_dir, 'modality_cluster_mappings.pkl'), 'wb') as f:
            pickle.dump(modality_cluster_mappings, f)


def load_dataloader(save_dir: str) -> DataLoader:
    """
    从保存的配置重建数据加载器

    参数:
        save_dir: 保存目录路径

    返回:
        重建的DataLoader实例
    """
    # 1. 加载数据加载器配置
    with open(os.path.join(save_dir, 'dataloader_config.json'), 'r') as f:
        dl_config = json.load(f)

    # 2. 加载嵌入向量
    embeddings_vectors = {}
    for fname in os.listdir(save_dir):
        if fname.endswith('_embeddings.pkl'):
            modality = fname.split('_embeddings.pkl')[0]
            embeddings_vectors[modality] = pd.read_pickle(os.path.join(save_dir, fname))

    # 3. 加载基因映射
    with open(os.path.join(save_dir, 'gene_mappings.pkl'), 'rb') as f:
        embeddings_genes = pickle.load(f)

    # 4. 尝试加载样本列表
    samples_path = os.path.join(save_dir, 'samples.pkl')
    if os.path.exists(samples_path):
        with open(samples_path, 'rb') as f:
            samples = pickle.load(f)
    else:
        samples = None

    # 5. 尝试加载聚类信息
    cluster_path = os.path.join(save_dir, 'modality_cluster_mappings.pkl')
    if os.path.exists(cluster_path):
        with open(cluster_path, 'rb') as f:
            modality_cluster_mappings = pickle.load(f)
    else:
        modality_cluster_mappings = None

    # 6. 重建数据加载器
    return create_resampling_dataloader(
        embeddings_vectors=embeddings_vectors,
        embeddings_genes=embeddings_genes,
        batch_size=dl_config['batch_size'],
        shuffle=dl_config['shuffle'],
        samples=samples,
        modality_cluster_mappings=modality_cluster_mappings
    )


# 使用示例
def main():
    # 基础路径
    base_path = 'data/species/Homo_sapiens'

    # 定义文件路径和对应的模态
    embeddings_files = {
        'DNA': 'embeddings/DNA_merged_embeddings.csv.gz',
        'RNA': 'embeddings/LucaOne_RNA_merged_embeddings.csv.gz',
        'protein': 'embeddings/proteins_merged_embeddings.csv.gz',
        'text': 'embeddings/gene_summary_text_embeddings.csv.gz',
        'singlecell': 'embeddings/scGPT_gene_embeddings.csv.gz'
    }

    csv_files = {
        'DNA': 'cds_data.csv.gz',
        'RNA': 'rna_data.csv.gz',
        'protein': 'protein_data.csv.gz',
        'text': None,
        'singlecell': None
    }

    # 定义每个模态的聚类数量
    n_clusters_per_modality = {
        'DNA': 20,
        'RNA': 20,
        'protein': 20,
        'text': 20,
        'singlecell': 20
    }

    # 加载数据
    try:
        embeddings_vectors, embeddings_genes = load_embeddings_and_gene_mappings(base_path, embeddings_files, csv_files)

        # 打印各模态信息
        for modality in embeddings_vectors:
            print(f"模态: {modality}")
            print(f"嵌入数据形状: {embeddings_vectors[modality].shape}")
            print(f"前5个基因名称: {embeddings_genes[modality][:5]}")
            print("-" * 50)

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return

    # 执行模态聚类
    print("开始模态聚类...")
    modality_clusters = perform_modality_clustering(embeddings_vectors, n_clusters_per_modality)

    # 创建各模态的聚类映射
    print("\n创建各模态的聚类映射...")
    modality_cluster_mappings = create_modality_cluster_mappings(embeddings_genes, modality_clusters)

    # 创建完整数据集获取所有样本组合
    full_dataset = ResamplingMultiModalGeneDatasetNotAll2(
        embeddings_vectors,
        embeddings_genes,
        modality_cluster_mappings=modality_cluster_mappings
    )
    all_samples = full_dataset.samples

    # 划分训练集、验证集和测试集 (70:15:15)
    # 首先划分训练集和临时集 (70:30)
    train_samples, temp_samples = split_samples(all_samples, train_ratio=0.7)
    # 然后将临时集平分为验证集和测试集 (各15%)
    val_samples, test_samples = split_samples(temp_samples, train_ratio=0.5)

    batch_size = 512
    # 创建训练、验证和测试数据加载器
    train_dataloader = create_resampling_dataloader(
        embeddings_vectors,
        embeddings_genes,
        batch_size=batch_size,
        shuffle=True,
        samples=train_samples,
        modality_cluster_mappings=modality_cluster_mappings
    )

    val_dataloader = create_resampling_dataloader(
        embeddings_vectors,
        embeddings_genes,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        samples=val_samples,
        modality_cluster_mappings=modality_cluster_mappings
    )

    test_dataloader = create_resampling_dataloader(
        embeddings_vectors,
        embeddings_genes,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        samples=test_samples,
        modality_cluster_mappings=modality_cluster_mappings
    )

    # 测试训练数据加载器
    print("\n训练数据加载器测试:")
    train_gene_tracker = defaultdict(int)

    for i, batch in enumerate(train_dataloader):
        print(f"\n训练Batch {i}:")
        print(f"基因数量: {len(batch['genes'])}")
        print(f"cluster字段类型: {type(batch['cluster'])}")
        print(f"cluster字段包含的模态: {list(batch['cluster'].keys())}")

        # 打印各模态的聚类标签分布
        for mod, cluster_tensor in batch['cluster'].items():
            cluster_labels = cluster_tensor.numpy()
            print(f"{mod}模态聚类标签分布: {np.bincount(cluster_labels[cluster_labels >= 0])}")

        # 跟踪基因出现次数
        for gene in batch['genes']:
            train_gene_tracker[gene] += 1

        print_dict_info(batch)

        if i == 2:  # 只查看前3个batch
            break

    # 保存训练、验证和测试数据加载器配置
    train_save_dir = 'data/dataloader/Homo_sapiens-M5-ModalityClusters-V17-C20/train'
    save_dataloader_config(
        train_dataloader,
        embeddings_vectors,
        embeddings_genes,
        train_save_dir,
        samples=train_samples,
        modality_cluster_mappings=modality_cluster_mappings
    )
    print(f"\n已保存训练数据加载器配置到 {train_save_dir}")

    val_save_dir = 'data/dataloader/Homo_sapiens-M5-ModalityClusters-V17-C20/val'
    save_dataloader_config(
        val_dataloader,
        embeddings_vectors,
        embeddings_genes,
        val_save_dir,
        samples=val_samples,
        modality_cluster_mappings=modality_cluster_mappings
    )
    print(f"已保存验证数据加载器配置到 {val_save_dir}")

    test_save_dir = 'data/dataloader/Homo_sapiens-M5-ModalityClusters-V17-C20/test'
    save_dataloader_config(
        test_dataloader,
        embeddings_vectors,
        embeddings_genes,
        test_save_dir,
        samples=test_samples,
        modality_cluster_mappings=modality_cluster_mappings
    )
    print(f"已保存测试数据加载器配置到 {test_save_dir}")

    # 测试加载保存的数据加载器
    print("\n从保存的配置重建的数据加载器:")
    loaded_train_dataloader = load_dataloader(train_save_dir)

    # 测试重建的训练数据加载器
    for i, batch in enumerate(loaded_train_dataloader):
        print(f"\n重建的训练Batch {i}:")
        print(f"基因数量: {len(batch['genes'])}")
        print(f"cluster字段类型: {type(batch['cluster'])}")
        print(f"cluster字段包含的模态: {list(batch['cluster'].keys())}")

        print_dict_info(batch)

        if i == 2:  # 只查看2个batch
            break


def print_dict_info(batch):
    for key, value in batch.items():
        # 获取数据类型
        data_type = type(value)

        if key == 'cluster':
            # 特殊处理cluster字段
            print(f"Key: {key}")
            print(f"  Type: {data_type}")
            print(f"  Structure: 字典包含以下模态: {list(value.keys())}")
            for mod, tensor in value.items():
                print(f"    {mod}: shape {tensor.shape}")
        else:
            # 获取形状（如果是numpy数组或torch张量）
            if hasattr(value, 'shape'):
                shape = value.shape
            elif hasattr(value, 'size'):  # 对于PyTorch张量
                shape = value.size()
            else:
                shape = "N/A (not an array/tensor)"

            print(f"Key: {key}")
            print(f"  Type: {data_type}")
            print(f"  Shape: {shape}")
        print()


if __name__ == '__main__':
    main()