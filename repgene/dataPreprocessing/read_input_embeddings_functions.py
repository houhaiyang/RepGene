import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pickle
import json

# 设置随机种子保证可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


class ResamplingMultiModalGeneDataset(Dataset):
    def __init__(self, embeddings_vectors: Dict[str, pd.DataFrame],
                 embeddings_genes: Dict[str, List[str]],
                 samples: Optional[List[Dict[str, Tuple[str, int]]]] = None):
        """
        支持重采样的多模态基因数据集，支持传入样本列表

        参数:
            embeddings_vectors: 各模态的嵌入向量字典
            embeddings_genes: 各模态的基因名称列表字典
            samples: 可选的样本列表，如果为None则创建所有样本
        """
        self.modalities = list(embeddings_vectors.keys())
        self.embeddings = embeddings_vectors

        # 构建基因到各模态索引的完整映射
        self.gene_to_indices = self._build_complete_gene_index_mapping(embeddings_genes)

        # 创建可重采样样本列表
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._create_resampling_samples()

        # 记录每个基因的嵌入使用情况
        self.used_embeddings = defaultdict(set)

    def _build_complete_gene_index_mapping(self, embeddings_genes: Dict[str, List[str]]) -> Dict[
        str, Dict[str, List[int]]]:
        """
        构建完整的基因索引映射，包含所有基因的所有嵌入

        返回:
            字典结构: {gene: {modality: [index1, index2,...]}}
        """
        gene_dict = defaultdict(lambda: defaultdict(list))

        for mod in self.modalities:
            for idx, gene in enumerate(embeddings_genes[mod]):
                gene_dict[gene][mod].append(idx)

        return dict(gene_dict)

    def _create_resampling_samples(self) -> List[Dict[str, Tuple[str, int]]]:
        """
        创建可重采样的样本列表，每个样本包含基因和各模态的嵌入索引

        返回:
            列表结构: [{'gene': gene, 'DNA': (gene, idx), ...}, ...]
        """
        samples = []

        for gene, mod_indices in self.gene_to_indices.items():
            # 检查基因是否在所有模态都有至少一个嵌入
            if all(len(indices) > 0 for indices in mod_indices.values()):
                # 创建该基因的所有嵌入组合
                from itertools import product

                # 获取各模态的嵌入索引列表
                mod_indices_lists = [mod_indices[mod] for mod in self.modalities]

                # 生成所有可能的嵌入组合
                for embedding_comb in product(*mod_indices_lists):
                    sample = {'gene': gene}
                    for mod, idx in zip(self.modalities, embedding_comb):
                        sample[mod] = (gene, idx)  # 存储基因和嵌入索引
                    samples.append(sample)

        print(f"创建了 {len(samples)} 个可重采样样本")
        return samples

    def __len__(self) -> int:
        """返回可重采样样本的总数"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本的各模态嵌入"""
        sample_info = self.samples[idx]
        gene = sample_info['gene']
        sample = {'gene': gene}

        # 记录使用的嵌入
        for mod in self.modalities:
            _, embed_idx = sample_info[mod]
            self.used_embeddings[gene].add(embed_idx)

        # 获取各模态的嵌入向量
        for mod in self.modalities:
            gene, embed_idx = sample_info[mod]
            embedding = self.embeddings[mod].iloc[embed_idx].values
            sample[mod] = torch.FloatTensor(embedding)

        return sample


class ResamplingMultiModalGeneDatasetNotAll(Dataset):
    def __init__(self, embeddings_vectors: Dict[str, pd.DataFrame],
                 embeddings_genes: Dict[str, List[str]],
                 samples: Optional[List[Dict[str, Tuple[str, int]]]] = None):
        """
        新的多模态基因数据集，确保每个嵌入至少使用一次，而不是创建所有组合

        参数:
            embeddings_vectors: 各模态的嵌入向量字典
            embeddings_genes: 各模态的基因名称列表字典
            samples: 可选的样本列表，如果为None则创建优化样本
        """
        self.modalities = list(embeddings_vectors.keys())
        self.embeddings = embeddings_vectors

        # 构建基因到各模态索引的完整映射
        self.gene_to_indices = self._build_complete_gene_index_mapping(embeddings_genes)

        # 创建优化样本列表
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._create_optimized_samples()

        # 记录每个基因的嵌入使用情况
        self.used_embeddings = defaultdict(set)

    def _build_complete_gene_index_mapping(self, embeddings_genes: Dict[str, List[str]]) -> Dict[
        str, Dict[str, List[int]]]:
        """
        构建完整的基因索引映射（与原始版本相同）
        """
        gene_dict = defaultdict(lambda: defaultdict(list))

        for mod in self.modalities:
            for idx, gene in enumerate(embeddings_genes[mod]):
                gene_dict[gene][mod].append(idx)

        return dict(gene_dict)

    def _create_optimized_samples(self) -> List[Dict[str, Tuple[str, int]]]:
        """
        创建优化样本，确保每个嵌入至少使用一次
        样本数 = 所有模态的总嵌入数（大大少于全组合）
        """
        samples = []
        embedding_usage = {mod: set() for mod in self.modalities}  # 跟踪每个模态的嵌入使用情况

        # 首先处理所有基因的嵌入
        for gene, mod_indices in self.gene_to_indices.items():
            for mod in self.modalities:
                for idx in mod_indices[mod]:
                    # 如果这个嵌入尚未使用，创建新样本
                    if idx not in embedding_usage[mod]:
                        sample = {'gene': gene}

                        # 对于当前模态，使用特定嵌入
                        sample[mod] = (gene, idx)
                        embedding_usage[mod].add(idx)  # 标记为已使用

                        # 对于其他模态，使用随机嵌入（如果可用）
                        for other_mod in self.modalities:
                            if other_mod == mod:
                                continue

                            if mod_indices[other_mod]:
                                # 随机选择一个未使用的嵌入（优先）或任意嵌入
                                other_idx = self._select_optimal_embedding(
                                    mod_indices[other_mod],
                                    embedding_usage[other_mod]
                                )
                                sample[other_mod] = (gene, other_idx)
                                embedding_usage[other_mod].add(other_idx)

                        samples.append(sample)

        # 处理剩余未使用的嵌入
        for mod in self.modalities:
            for idx in set(range(len(self.embeddings[mod]))) - embedding_usage[mod]:
                # 创建样本只包含这个未使用的嵌入
                sample = {'gene': None}
                sample[mod] = (None, idx)
                embedding_usage[mod].add(idx)

                # 其他模态留空（实际使用时需处理）
                for other_mod in self.modalities:
                    if other_mod != mod:
                        sample[other_mod] = (None, -1)  # 特殊标记
                samples.append(sample)

        print(f"优化样本创建完成: 共 {len(samples)} 个样本（远少于全组合）")
        return samples

    def _select_optimal_embedding(self, indices: List[int], used_set: set) -> int:
        """
        优先选择未使用的嵌入，如果都已使用则随机选择
        """
        unused = [idx for idx in indices if idx not in used_set]
        if unused:
            return random.choice(unused)
        return random.choice(indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本并记录嵌入使用情况"""
        sample_info = self.samples[idx]
        gene = sample_info['gene']
        sample = {'gene': gene}

        # 处理每个模态的嵌入
        for mod in self.modalities:
            gene_id, embed_idx = sample_info[mod]

            # 跳过特殊标记的无效嵌入
            if embed_idx == -1 or gene_id is None:
                sample[mod] = torch.zeros(self.embeddings[mod].shape[1], dtype=torch.float)
                continue

            # 记录使用情况
            self.used_embeddings[gene].add(embed_idx)

            # 获取嵌入向量
            embedding = self.embeddings[mod].iloc[embed_idx].values
            sample[mod] = torch.FloatTensor(embedding)

        return sample


class ResamplingMultiModalGeneDatasetNotAll2(Dataset):
    def __init__(self, embeddings_vectors: Dict[str, pd.DataFrame],
                 embeddings_genes: Dict[str, List[str]],
                 samples: Optional[List[Dict[str, Tuple[str, int]]]] = None):
        """
        新的多模态基因数据集，确保每个基因的每个模态序列至少使用一次

        参数:
            embeddings_vectors: 各模态的嵌入向量字典
            embeddings_genes: 各模态的基因名称列表字典
            samples: 可选的样本列表，如果为None则创建优化样本
        """
        self.modalities = list(embeddings_vectors.keys())
        self.embeddings = embeddings_vectors
        self.embeddings_genes = embeddings_genes

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        sample_info = self.samples[idx]
        gene = sample_info['gene']
        sample = {'gene': gene}

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
        }
    """
    collated = {'genes': [item['gene'] for item in batch]}

    # 获取第一个样本的所有模态键（排除'gene'键）
    modalities = [key for key in batch[0].keys() if key != 'gene']

    for mod in modalities:
        # 堆叠各模态的嵌入
        collated[mod] = torch.stack([item[mod] for item in batch])

    return collated


def create_resampling_dataloader(embeddings_vectors: Dict[str, pd.DataFrame],
                                 embeddings_genes: Dict[str, List[str]],
                                 batch_size: int = 512,
                                 shuffle: bool = True,
                                 samples: List[Dict] = None) -> DataLoader:
    """
    创建支持重采样的多模态基因数据加载器

    参数:
        embeddings_vectors: 各模态的嵌入向量字典
        embeddings_genes: 各模态的基因名称列表字典
        batch_size: 批量大小
        shuffle: 是否打乱数据
        samples: 可选样本列表

    返回:
        DataLoader实例
    """
    dataset = ResamplingMultiModalGeneDataset(embeddings_vectors, embeddings_genes, samples=samples)
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
                           samples: List[Dict] = None):
    """
    保存重建数据加载器所需的配置，包括样本列表

    参数:
        dataloader: 要保存的数据加载器
        embeddings_vectors: 各模态的嵌入向量字典
        embeddings_genes: 各模态的基因名称列表字典
        save_dir: 保存目录路径
        samples: 样本列表
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

    # 5. 重建数据加载器
    return create_resampling_dataloader(
        embeddings_vectors=embeddings_vectors,
        embeddings_genes=embeddings_genes,
        batch_size=dl_config['batch_size'],
        shuffle=dl_config['shuffle'],
        samples=samples
    )



