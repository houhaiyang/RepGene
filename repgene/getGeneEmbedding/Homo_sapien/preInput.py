import os
import re
import pandas as pd
from Bio import SeqIO


def parse_cds_description(description):
    """解析CDS描述中的结构化信息"""
    info = {
        'gene': None,
        'protein': None,
        'protein_id': None,
        'location': None,
        'db_xref': {}
    }

    # 提取gene
    gene_match = re.search(r'\[gene=([^\]]+)\]', description)
    if gene_match:
        info['gene'] = gene_match.group(1)

    # 提取protein
    protein_match = re.search(r'\[protein=([^\]]+)\]', description)
    if protein_match:
        info['protein'] = protein_match.group(1)

    # 提取protein_id
    protein_id_match = re.search(r'\[protein_id=([^\]]+)\]', description)
    if protein_id_match:
        info['protein_id'] = protein_id_match.group(1)

    # 提取location
    location_match = re.search(r'\[location=([^\]]+)\]', description)
    if location_match:
        info['location'] = location_match.group(1)

    # 提取db_xref
    db_xref_match = re.search(r'\[db_xref=([^\]]+)\]', description)
    if db_xref_match:
        xrefs = db_xref_match.group(1).split(',')
        for xref in xrefs:
            if ':' in xref:
                key, value = xref.split(':', 1)
                info['db_xref'][key] = value

    return info

def parse_rna_description(description):
    """解析RNA描述中的结构化信息"""
    info = {
        'gene': None,
        'gene_symbol': None,
        'transcript_variant': None,
        'protein_product': None,
        'mRNA_id': None
    }

    # 示例描述: "NM_000014.6 Homo sapiens alpha-2-macroglobulin (A2M), transcript variant 1, mRNA"

    # 提取基因符号 (括号中的内容)
    gene_symbol_match = re.search(r'\(([^)]+)\)', description)
    if gene_symbol_match:
        info['gene_symbol'] = gene_symbol_match.group(1)

    # 提取蛋白质产品名称
    protein_match = re.search(r'([^(]+)\s*\(', description)
    if protein_match:
        info['protein_product'] = protein_match.group(1).strip()

    # 提取转录本变体
    variant_match = re.search(r'transcript variant (\d+)', description)
    if variant_match:
        info['transcript_variant'] = variant_match.group(1)

    # mRNA ID通常是记录ID本身
    info['mRNA_id'] = description.split()[0]

    return info

def parse_protein_description(description):
    """解析蛋白质描述中的结构化信息"""
    info = {
        'protein_id': None,
        'protein_name': None,
        'isoform': None,
        'organism': None
    }

    # 示例描述: "NP_000005.3 alpha-2-macroglobulin isoform a precursor [Homo sapiens]"

    # 提取蛋白质名称
    parts = description.split('[')[0].split()
    if len(parts) > 1:
        info['protein_id'] = parts[0]
        info['protein_name'] = ' '.join(parts[1:])

    # 提取异构体信息
    isoform_match = re.search(r'isoform (\w+)', description)
    if isoform_match:
        info['isoform'] = isoform_match.group(1)

    # 提取生物体
    organism_match = re.search(r'\[([^]]+)\]', description)
    if organism_match:
        info['organism'] = organism_match.group(1)

    return info



def cds_records_to_dataframe(cds_records):
    """将CDS记录列表转换为DataFrame"""
    data = []

    for record in cds_records:
        # 解析基本信息
        record_info = {
            'id': record.id,
            'description': record.description,
            'sequence_length': len(record.seq),
            'sequence': str(record.seq)  # 存储完整序列
        }

        # 解析结构化描述信息
        desc_info = parse_cds_description(record.description)
        record_info.update(desc_info)

        # 添加db_xref的单独字段
        for key, value in desc_info['db_xref'].items():
            record_info[f'db_xref_{key}'] = value

        data.append(record_info)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 调整列顺序（移除了sequence_start，添加了sequence）
    cols = ['id', 'sequence', 'gene', 'protein', 'protein_id',
            'sequence_length', 'sequence',
            'description', 'location',
            'db_xref_CCDS', 'db_xref_Ensembl', 'db_xref_GeneID']

    # 只保留存在的列
    existing_cols = [col for col in cols if col in df.columns]
    df = df[existing_cols + [col for col in df.columns if col not in existing_cols]]

    return df

def rna_records_to_dataframe(rna_records):
    """将RNA记录列表转换为DataFrame"""
    data = []

    for record in rna_records:
        record_info = {
            'id': record.id,
            'description': record.description,
            'sequence_length': len(record.seq),
            'sequence': str(record.seq)
        }

        # 解析结构化描述信息
        desc_info = parse_rna_description(record.description)
        record_info.update(desc_info)

        data.append(record_info)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 调整列顺序
    cols = ['id', 'sequence', 'gene_symbol', 'protein_product', 'transcript_variant',
            'sequence_length', 'description', 'mRNA_id']

    # 只保留存在的列
    existing_cols = [col for col in cols if col in df.columns]
    df = df[existing_cols + [col for col in df.columns if col not in existing_cols]]

    return df

def protein_records_to_dataframe(protein_records):
    """将蛋白质记录列表转换为DataFrame"""
    data = []

    for record in protein_records:
        record_info = {
            'id': record.id,
            'description': record.description,
            'sequence_length': len(record.seq),
            'sequence': str(record.seq)
        }

        # 解析结构化描述信息
        desc_info = parse_protein_description(record.description)
        record_info.update(desc_info)

        data.append(record_info)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 调整列顺序
    cols = ['id', 'sequence', 'protein_name', 'isoform', 'organism',
            'sequence_length', 'description', 'protein_id']

    # 只保留存在的列
    existing_cols = [col for col in cols if col in df.columns]
    df = df[existing_cols + [col for col in df.columns if col not in existing_cols]]

    return df


# 定义文件路径
species = 'Homo_sapiens'
# 基础路径部分
base_dir = f"data/species/{species}/{species}/ncbi_dataset/data"
# 查找GCF开头的文件夹
gcf_folders = [f for f in os.listdir(base_dir) if f.startswith("GCF")]
base_path = os.path.join(base_dir, gcf_folders[0])
data_dir = f'data/species/{species}'
# 处理 lcl| 前缀 ---------------------------
# 定义文件路径
input_file = os.path.join(base_path, "cds_from_genomic.fna")
output_file = os.path.join(base_path, "cds_from_genomic_1.fna")
records = list(SeqIO.parse(input_file, "fasta"))
for record in records:
    # 处理ID
    if record.id.startswith("lcl|"):
        record.id = record.id[4:]
    if record.name.startswith("lcl|"):
        record.name = record.id[4:]
    # 处理描述
    if record.description.startswith("lcl|"):
        record.description = record.description[4:]
# 保存处理后的文件
with open(output_file, "w") as output_handle:
    SeqIO.write(records, output_handle, "fasta")


# 定义三个文件路径
cds_file = os.path.join(base_path, "cds_from_genomic_1.fna")
protein_file = os.path.join(base_path, "protein.faa")
rna_file = os.path.join(base_path, "rna.fna")

def read_sequences(file_path, file_type):
    """读取序列文件的通用函数"""
    records = list(SeqIO.parse(file_path, file_type))
    print(f"从 {os.path.basename(file_path)} 中读取了 {len(records)} 条序列")
    return records

# 读取CDS序列 (FASTA格式)
cds_records = read_sequences(cds_file, "fasta")
# 读取RNA序列 (FASTA格式)
rna_records = read_sequences(rna_file, "fasta")
# 读取蛋白质序列 (FASTA格式)
protein_records = read_sequences(protein_file, "fasta")


# 示例：打印第一条CDS记录的信息
if cds_records:
    print("\n第一条CDS记录示例:")
    print(f"ID: {cds_records[0].id}")
    print(f"描述: {cds_records[0].description}")
    print(f"序列长度: {len(cds_records[0].seq)}")
    print(f"前60个碱基: {cds_records[0].seq[:60]}...")

# 示例：打印第一条RNA记录的信息
if rna_records:
    print("\n第一条RNA记录示例:")
    print(f"ID: {rna_records[0].id}")
    print(f"描述: {rna_records[0].description}")
    print(f"序列长度: {len(rna_records[0].seq)}")
    print(f"前30个氨基酸: {rna_records[0].seq[:30]}...")

# 示例：打印第一条蛋白质记录的信息
if protein_records:
    print("\n第一条蛋白质记录示例:")
    print(f"ID: {protein_records[0].id}")
    print(f"描述: {protein_records[0].description}")
    print(f"序列长度: {len(protein_records[0].seq)}")
    print(f"前30个氨基酸: {protein_records[0].seq[:30]}...")


# 转换为DataFrame
cds_df = cds_records_to_dataframe(cds_records)
rna_df = rna_records_to_dataframe(rna_records)
protein_df = protein_records_to_dataframe(protein_records)

rna_df['gene'] = rna_df['gene_symbol']
# 首先创建一个protein_id到gene的映射字典
protein_to_gene = cds_df.dropna(subset=['protein_id']).drop_duplicates('protein_id').set_index('protein_id')['gene'].to_dict()
# 将映射应用到protein_df
protein_df['gene'] = protein_df['protein_id'].map(protein_to_gene)
# 对cds_df按sequence_length升序排序
cds_df = cds_df.sort_values(by='sequence_length', ascending=True)
# 查看排序结果
print(cds_df[['id', 'sequence_length']].head(5))  # 显示最短的5条序列
print(cds_df[['id', 'sequence_length']].tail(5))  # 显示最长的5条序列
count_long_sequences = (cds_df['sequence_length'] > 27000).sum()
print(f"长度 > 27000 的序列数量: {count_long_sequences}")


# ------------------------------- 统计数据 ------------------------------- #
# 统计CDS数据
if not cds_df.empty:
    unique_cds = cds_df['id'].nunique()
    print(f"\n唯一CDS数量: {unique_cds}")

    if 'gene' in cds_df.columns:
        unique_genes = cds_df['gene'].nunique()
        print(f"唯一Gene数量: {unique_genes}")

    if 'protein' in cds_df.columns:
        unique_proteins = cds_df['protein_id'].nunique()
        print(f"唯一Protein id数量: {unique_proteins}")

    print("\nCDS记录示例:")
    print(cds_df.head())

# 统计RNA数据
if not rna_df.empty:
    unique_rnas = rna_df['id'].nunique()
    print(f"\n唯一RNA id数量: {unique_rnas}")

    if 'gene' in rna_df.columns:
        unique_genes = rna_df['gene'].nunique()
        print(f"唯一Gene数量: {unique_genes}")

    print("\nRNA记录示例:")
    print(rna_df.head())

# 统计蛋白质数据
if not protein_df.empty:
    unique_proteins = protein_df['id'].nunique()
    print(f"\n唯一Protein id数量: {unique_proteins}")

    if 'protein_name' in protein_df.columns:
        unique_protein_names = protein_df['protein_name'].nunique()
        print(f"唯一Protein name数量: {unique_protein_names}")

    if 'gene' in protein_df.columns:
        unique_genes = protein_df['gene'].nunique()
        print(f"唯一Gene数量: {unique_genes}")

    print("\n蛋白质记录示例:")
    print(protein_df.head())


# 保存 CDS 数据
cds_df.to_csv(os.path.join(data_dir, "cds_data.csv.gz"), index=False, compression='gzip')
# 保存 RNA 数据
rna_df.to_csv(os.path.join(data_dir, "rna_data.csv.gz"), index=False, compression='gzip')
# 保存 Protein 数据
protein_df.to_csv(os.path.join(data_dir, "protein_data.csv.gz"), index=False, compression='gzip')
print(f"数据已保存到: {data_dir}")


# print(cds_df.columns)
# print(rna_df.columns)
# print(protein_df.columns)

# 获取三个数据框的gene集合
cds_genes = set(cds_df['gene'].dropna().unique())
rna_genes = set(rna_df['gene'].dropna().unique())
protein_genes = set(protein_df['gene'].dropna().unique())

# 计算交集 gene（共有的 gene）
common_genes = cds_genes & rna_genes & protein_genes
common_genes_count = len(common_genes)
# 计算并集 gene（所有出现过的 gene）
union_genes = cds_genes | rna_genes | protein_genes
union_genes_count = len(union_genes)

print(f"\nCDS 中唯一 gene 数量: {len(cds_genes)}")
print(f"RNA 中唯一 gene 数量: {len(rna_genes)}")
print(f"Protein 中唯一 gene 数量: {len(protein_genes)}")
print(f"三个数据框共有的 gene（交集）数量: {common_genes_count}")
print(f"三个数据框所有出现过的 gene（并集）数量: {union_genes_count}")


