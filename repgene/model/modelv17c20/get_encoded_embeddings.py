
# 1.只考虑在所有模态中都出现的基因（即每个模态都有该基因的嵌入）
# 2.对于这些基因，我们使用所有模态的嵌入，通过fusion模块得到融合嵌入，然后保存。
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Set
import json
import tempfile
import shutil
import pickle
import os
from typing import Dict, Set
import numpy as np

# 从新模型文件中导入必要的定义
from repgene.model.modelv17c20.repgene_model import RepGeneV17

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_embeddings_and_gene_mappings(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    从数据加载器目录加载嵌入向量和基因映射

    参数:
        data_dir: 数据加载器目录路径

    返回:
        embeddings_vectors: 各模态的嵌入向量字典
        gene_mappings: 各模态序列ID到基因名称的映射字典
    """
    embeddings_vectors = {}
    gene_mappings = {}

    # 加载嵌入向量
    for fname in os.listdir(data_dir):
        if fname.endswith('_embeddings.pkl'):
            modality = fname.split('_embeddings.pkl')[0]
            with open(os.path.join(data_dir, fname), 'rb') as f:
                embeddings_vectors[modality] = pd.read_pickle(f)

    # 加载基因映射
    with open(os.path.join(data_dir, 'gene_mappings.pkl'), 'rb') as f:
        gene_mappings = pickle.load(f)

    return embeddings_vectors, gene_mappings


def find_common_genes(gene_mappings: Dict[str, List[str]]) -> Set[str]:
    """
    找出所有模态都存在的基因

    参数:
        gene_mappings: 各模态的基因名称列表字典

    返回:
        所有模态都存在的基因集合
    """
    if not gene_mappings:
        return set()

    # 取第一个模态的基因集合作为初始集合
    common_genes = set(gene_mappings[list(gene_mappings.keys())[0]])

    # 与其他模态取交集
    for modality, genes in gene_mappings.items():
        common_genes = common_genes.intersection(set(genes))

    print(f"所有模态都存在的基因数量: {len(common_genes)}")
    return common_genes


def filter_embeddings_by_common_genes(
        embeddings_vectors: Dict[str, pd.DataFrame],
        gene_mappings: Dict[str, List[str]],
        common_genes: Set[str]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    根据共有基因筛选嵌入向量和基因映射

    参数:
        embeddings_vectors: 各模态的嵌入向量字典
        gene_mappings: 各模态的基因名称列表字典
        common_genes: 共有基因集合

    返回:
        筛选后的嵌入向量和基因映射
    """
    filtered_embeddings = {}
    filtered_mappings = {}

    for modality in embeddings_vectors.keys():
        # 获取当前模态的基因列表
        genes = gene_mappings[modality]

        # 创建掩码，标记哪些基因是共有的
        mask = [gene in common_genes for gene in genes]

        # 应用筛选
        filtered_embeddings[modality] = embeddings_vectors[modality].iloc[mask]
        filtered_mappings[modality] = [genes[i] for i in range(len(genes)) if mask[i]]

        print(f"模态 {modality}: 筛选前 {len(genes)} 个基因, 筛选后 {len(filtered_mappings[modality])} 个基因")

    return filtered_embeddings, filtered_mappings


class CommonGenesDataset(Dataset):
    """用于处理所有模态都存在的基因的数据集"""

    def __init__(self, embeddings: Dict[str, pd.DataFrame], gene_mappings: Dict[str, List[str]],
                 common_genes: Set[str]):
        """
        参数:
            embeddings: 各模态的嵌入向量字典
            gene_mappings: 各模态的基因名称列表字典
            common_genes: 所有模态都存在的基因集合
        """
        self.embeddings = embeddings
        self.gene_mappings = gene_mappings
        self.common_genes = sorted(list(common_genes))
        self.modalities = list(embeddings.keys())

        # 构建基因到各模态嵌入索引的映射
        self.gene_to_indices = {}
        for gene in self.common_genes:
            self.gene_to_indices[gene] = {}
            for mod in self.modalities:
                # 找到该基因在当前模态中的所有索引
                indices = [i for i, g in enumerate(gene_mappings[mod]) if g == gene]
                if not indices:
                    # 理论上不会发生，因为我们已经筛选了共有基因
                    raise ValueError(f"基因 {gene} 在模态 {mod} 中不存在")
                self.gene_to_indices[gene][mod] = indices

    def __len__(self):
        return len(self.common_genes)

    def __getitem__(self, idx):
        gene = self.common_genes[idx]
        sample = {'gene': gene}

        # 为每个模态选择第一条嵌入（而不是随机选择）
        for mod in self.modalities:
            indices = self.gene_to_indices[gene][mod]
            # 选择第一条嵌入，确保结果可重现
            embed_idx = indices[0]
            embedding = self.embeddings[mod].iloc[embed_idx].values
            sample[mod] = torch.FloatTensor(embedding)

        return sample


def compute_common_genes_embeddings(
        model: RepGeneV17,
        embeddings_vectors: Dict[str, pd.DataFrame],
        gene_mappings: Dict[str, List[str]],
        common_genes: Set[str],
        batch_size: int = 512
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    计算所有模态都存在的基因的嵌入表示

    参数:
        model: 训练好的模型
        embeddings_vectors: 各模态的嵌入向量字典
        gene_mappings: 各模态的基因名称列表字典
        common_genes: 所有模态都存在的基因集合
        batch_size: 批量大小

    返回:
        字典包含以下嵌入:
        {
            'raw': {
                'DNA': {gene: embedding},
                'RNA': {gene: embedding},
                ...
            },
            'adjusted': {
                'DNA': {gene: embedding},
                'RNA': {gene: embedding},
                ...
            },
            'encoded': {
                'DNA': {gene: embedding},
                'RNA': {gene: embedding},
                ...
            },
            'fused': {gene: embedding}  # 融合嵌入
        }
    """
    # 初始化存储结构
    results = {
        'raw': {mod: {} for mod in embeddings_vectors.keys()},
        'adjusted': {mod: {} for mod in embeddings_vectors.keys()},
        'encoded': {mod: {} for mod in embeddings_vectors.keys()}
    }

    # 创建数据集
    dataset = CommonGenesDataset(embeddings_vectors, gene_mappings, common_genes)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 模型推理
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算基因嵌入"):
            genes = batch['gene']

            # 准备输入数据
            inputs = {}
            for mod in model.modalities:
                if mod in batch:
                    inputs[mod] = batch[mod].to(device)

            # 模型推理
            outputs = model(inputs)

            # 保存原始嵌入 - 直接保存单条嵌入，不取平均
            for mod in model.modalities:
                if mod in batch:
                    raw_embeddings = batch[mod].cpu().numpy()
                    for i, gene in enumerate(genes):
                        # 直接保存单条嵌入，不进行平均
                        results['raw'][mod][gene] = raw_embeddings[i]

            # 保存adjusted和encoded输出 - 直接保存单条嵌入
            for mod in model.modalities:
                if mod in outputs['adjusted']:
                    adjusted_embeddings = outputs['adjusted'][mod].cpu().numpy()
                    encoded_embeddings = outputs['encoded'][mod].cpu().numpy()

                    for i, gene in enumerate(genes):
                        # 直接保存单条嵌入，不进行平均
                        results['adjusted'][mod][gene] = adjusted_embeddings[i]
                        results['encoded'][mod][gene] = encoded_embeddings[i]

    # 由于每个基因只保存了单条嵌入，不需要平均操作
    print("已完成基因嵌入计算（使用单条嵌入）")

    # 直接返回结果，不需要额外的处理
    embeddings = results

    return embeddings


class FusedEmbeddingDataset(Dataset):
    """用于批量计算融合嵌入的数据集"""

    def __init__(self, common_genes_embeddings: Dict[str, Dict[str, np.ndarray]],
                 modalities: List[str], common_genes: Set[str]):
        """
        参数:
            common_genes_embeddings: 共有基因的单模态嵌入结果
            modalities: 模态列表
            common_genes: 共有基因集合
        """
        self.common_genes = sorted(list(common_genes))
        self.modalities = modalities
        self.encoded_embeddings = common_genes_embeddings['encoded']

        # 验证所有基因在所有模态中都存在
        self.valid_genes = []
        for gene in self.common_genes:
            valid = True
            for mod in modalities:
                if gene not in self.encoded_embeddings[mod]:
                    valid = False
                    break
            if valid:
                self.valid_genes.append(gene)

        print(f"可用于融合嵌入计算的基因数量: {len(self.valid_genes)}")

    def __len__(self):
        return len(self.valid_genes)

    def __getitem__(self, idx):
        gene = self.valid_genes[idx]

        # 获取该基因在所有模态中的encoded嵌入
        modality_embeddings = {}
        for mod in self.modalities:
            modality_embeddings[mod] = torch.FloatTensor(self.encoded_embeddings[mod][gene])

        return gene, modality_embeddings


def compute_fused_embeddings_batch(
        model: RepGeneV17,
        common_genes_embeddings: Dict[str, Dict[str, np.ndarray]],
        common_genes: Set[str],
        batch_size: int = 512
) -> Dict[str, np.ndarray]:
    """
    批量计算融合嵌入（优化版本）

    参数:
        model: 训练好的模型
        common_genes_embeddings: 共有基因的单模态嵌入结果
        common_genes: 所有模态都存在的基因集合
        batch_size: 批量大小

    返回:
        融合嵌入字典: {gene: embedding}
    """
    # 创建数据集
    dataset = FusedEmbeddingDataset(common_genes_embeddings, model.modalities, common_genes)

    if len(dataset) == 0:
        print("警告: 没有可用于融合嵌入计算的基因")
        return {}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fused_batch
    )

    fused_embeddings = {}

    model.eval()
    with torch.no_grad():
        for batch_genes, batch_embeddings in tqdm(dataloader, desc="批量计算融合嵌入"):
            # 将嵌入移动到GPU
            gpu_embeddings = {}
            for mod in batch_embeddings:
                gpu_embeddings[mod] = batch_embeddings[mod].to(device)

            # 计算融合嵌入
            fused_batch = model.Fuser([gpu_embeddings[mod] for mod in model.modalities])

            # 保存结果
            fused_batch = fused_batch.cpu().numpy()
            for i, gene in enumerate(batch_genes):
                fused_embeddings[gene] = fused_batch[i]

    return fused_embeddings


def collate_fused_batch(batch):
    """
    自定义collate函数用于处理融合嵌入的批量数据
    """
    genes = [item[0] for item in batch]
    modality_embeddings_batch = [item[1] for item in batch]

    # 获取所有模态
    modalities = list(modality_embeddings_batch[0].keys())

    # 按模态组织数据
    batch_embeddings = {}
    for mod in modalities:
        # 堆叠该模态的所有嵌入
        mod_embeddings = [item[mod] for item in modality_embeddings_batch]
        batch_embeddings[mod] = torch.stack(mod_embeddings)

    return genes, batch_embeddings


def save_common_genes_embeddings(embeddings: dict, common_genes: Set[str], save_dir: str):
    """
    保存所有模态都存在的基因的嵌入结果到压缩的CSV文件 (.csv.gz)

    参数:
        embeddings: compute_common_genes_embeddings 返回的结果
        common_genes: 所有模态都存在的基因集合
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 只保存共有基因的嵌入
    common_genes_list = sorted(common_genes)

    # 保存各层嵌入
    layers_to_save = ['raw', 'adjusted', 'encoded']
    for layer in layers_to_save:
        for mod, gene_embeds in embeddings[layer].items():
            # 只保留共有基因
            filtered_embeds = {gene: gene_embeds[gene] for gene in common_genes_list if gene in gene_embeds}
            df = pd.DataFrame.from_dict(filtered_embeds, orient='index')
            df.index.name = 'gene'
            save_path = os.path.join(save_dir, f"{mod}_{layer}_embeddings.csv.gz")
            df.to_csv(save_path, compression='gzip')
            print(f"已保存 {mod} {layer}嵌入: {df.shape[0]} 个基因 -> {save_path}")

    # 保存融合嵌入（不分模态）
    if 'fused' in embeddings:
        # 只保留共有基因
        filtered_fused = {gene: embeddings['fused'][gene] for gene in common_genes_list if gene in embeddings['fused']}
        df = pd.DataFrame.from_dict(filtered_fused, orient='index')
        df.index.name = 'gene'
        save_path = os.path.join(save_dir, "fused_embeddings.csv.gz")
        df.to_csv(save_path, compression='gzip')
        print(f"已保存融合嵌入: {df.shape[0]} 个基因 -> {save_path}")


def loadBestModelWeights(model, model_dir):
    # 加载最佳模型权重
    model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"已加载最佳模型权重: {model_path}")
    else:
        # 如果找不到最佳模型，尝试加载最终模型
        model_path = os.path.join(model_dir, "final_model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"已加载最终模型权重: {model_path}")
        else:
            # 查找所有 epoch 模型文件
            import re
            model_files = [f for f in os.listdir(model_dir) if re.match(r"model_epoch\d+\.pth", f)]
            if model_files:
                # 提取 epoch 编号并找到最大的
                latest_file = max(model_files, key=lambda x: int(re.search(r"model_epoch(\d+)\.pth", x).group(1)))
                model_path = os.path.join(model_dir, latest_file)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                epoch_num = re.search(r"model_epoch(\d+)\.pth", latest_file).group(1)
                print(f"已加载最新 epoch 模型: {model_path} (epoch {epoch_num})")
            else:
                print("警告: 未找到任何模型权重文件，模型将使用随机初始化权重")

    return model


def load_config(config_path):
    """
    从指定路径加载配置文件

    参数:
        config_path (str): config.json文件的路径

    返回:
        dict: 包含配置信息的字典
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {config_path} 格式不正确")
        return None


def save_common_genes_embeddings_cache(
        common_genes_embeddings: Dict,
        common_genes: Set[str],
        cache_path: str
) -> None:
    """
    保存共有基因嵌入到缓存文件

    参数:
        common_genes_embeddings: 计算得到的共有基因嵌入
        common_genes: 共有基因集合
        cache_path: 缓存文件路径
    """
    # 创建缓存目录（如果不存在）
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 使用临时文件确保写入完整性
    temp_cache_path = cache_path + ".tmp"

    # 保存数据
    cache_data = {
        'common_genes_embeddings': common_genes_embeddings,
        'common_genes': common_genes
    }

    with open(temp_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    # 原子性地重命名文件
    os.rename(temp_cache_path, cache_path)
    print(f"已保存共有基因嵌入缓存: {cache_path}")

def load_common_genes_embeddings_cache(cache_path: str) -> tuple:
    """
    从缓存文件加载共有基因嵌入

    参数:
        cache_path: 缓存文件路径

    返回:
        tuple: (common_genes_embeddings, common_genes)
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    print(f"已加载共有基因嵌入缓存: {cache_path}")
    return cache_data['common_genes_embeddings'], cache_data['common_genes']


def main():
    # 配置路径
    data_dir = "data/dataloader/Homo_sapiens-M5-ModalityClusters/train"
    model_dir = "models/Homo_sapiens/2025-10-17-M5-V17"
    save_dir = os.path.join(model_dir, "embeddings")
    cache_dir = os.path.join(model_dir, "cache")  # 新增缓存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)  # 创建缓存目录
    cache_path = os.path.join(model_dir, "cache", "common_genes_embeddings.pkl")

    # 1. 加载嵌入数据和基因映射
    print("加载嵌入数据和基因映射...")
    embeddings_vectors, gene_mappings = load_embeddings_and_gene_mappings(data_dir)
    gene_mappings = {
        modality: [str(gene) for gene in genes]
        for modality, genes in gene_mappings.items()
    }

    # 2. 筛选所有模态都存在的基因
    print("\n筛选所有模态都存在的基因...")
    common_genes = find_common_genes(gene_mappings)

    # 3. 根据共有基因筛选嵌入数据
    embeddings_vectors, gene_mappings = filter_embeddings_by_common_genes(
        embeddings_vectors, gene_mappings, common_genes
    )

    # 统计各模态的非冗余基因集合
    unique_genes_per_modality = {
        modality: set(gene_list)
        for modality, gene_list in gene_mappings.items()
    }

    # 使用 pickle 保存（保持 set 类型）
    with open(f'{save_dir}/unique_genes_per_modality.pkl', 'wb') as f:
        pickle.dump(unique_genes_per_modality, f)

    # 打印各模态信息
    print("\n筛选后各模态嵌入维度:")
    for mod, df in embeddings_vectors.items():
        print(f"{mod}: {df.shape[0]} 个嵌入, 维度: {df.shape[1]}")

    # 4. 创建模型并加载权重
    print("\n创建模型并加载权重...")

    # 读取config文件
    config = load_config(os.path.join(model_dir, "config.json"))

    model = RepGeneV17(
        input_dims=config['input_dims'],
        encoder_type=config['encoder_type'],
        fusion_type=config['fusion_type'],
        decoder_type=config['decoder_type'],
        dropout_rate=config['dropout_rate'],
        n_clusters_per_modality=config['n_clusters_per_modality'],  # 修改参数名
        encoder_layers=config['encoder_layers'],
        fuser_layers=config['fuser_layers'],  # 修改参数名
        decoder_layers=config['decoder_layers']
    )

    model = loadBestModelWeights(model, model_dir).to(device)

    # 5. 计算共有基因的嵌入（使用缓存）
    print("\n计算所有模态都存在的基因的嵌入表示...")
    common_genes_embeddings = compute_common_genes_embeddings(
        model, embeddings_vectors, gene_mappings, common_genes,
        batch_size=config.get('batch_size', 512)
    )

    # 保存到缓存
    save_common_genes_embeddings_cache(common_genes_embeddings, common_genes, cache_path)
    # common_genes_embeddings, common_genes = load_common_genes_embeddings_cache(cache_path)

    # 6. 计算融合嵌入（优化版本）
    print("\n批量计算融合嵌入表示...")
    fused_embeddings = compute_fused_embeddings_batch(
        model, common_genes_embeddings, common_genes,
        batch_size=config.get('batch_size', 512)
    )

    common_genes_embeddings['fused'] = fused_embeddings

    # 7. 保存结果（只保存共有基因）
    print("\n保存嵌入结果...")
    save_common_genes_embeddings(common_genes_embeddings, common_genes, save_dir)
    print(f"\n所有嵌入已保存到: {save_dir}")


if __name__ == "__main__":
    main()
