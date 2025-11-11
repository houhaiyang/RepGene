#!/bin/bash
#DSUB -n ESM-C
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -q root.default
#DSUB -R "cpu=8;gpu=1;mem=24000"
#DSUB -pn "cyclone001-agent-155"
#DSUB -oo /home/share/huadjyin/home/houhaiyang/project/RepGene/logs/%J.out
#DSUB -eo /home/share/huadjyin/home/houhaiyang/project/RepGene/logs/%J.err


# 加载系统 Conda
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
# 加载环境配置
source /home/share/huadjyin/home/houhaiyang/bashrc/ESM-C-py310-torch241-cu118.bashrc

data_dir='/home/share/huadjyin/home/houhaiyang/project/RepGene/data/species/Homo_sapiens/'

# getEmbeddings
cd ${data_dir}
mkdir -p embeddings/proteins/
# 使用 ESM-C 生成蛋白序列的 Embeddings
python /home/share/huadjyin/home/houhaiyang/method/ESM-C/faa2emb_CLS-2.py \
    --input Homo_sapiens/ncbi_dataset/data/GCF_000001405.40/protein.faa \
    --outdir embeddings/proteins/

# mergeEmbeddings
cd ${data_dir}
python /home/share/huadjyin/home/houhaiyang/method/ESM-C/merge_emb.py \
    --input Homo_sapiens/ncbi_dataset/data/GCF_000001405.40/protein.faa \
    --embeddings_dir embeddings/proteins/ \
    --output ./embeddings/proteins_merged_embeddings.csv.gz

