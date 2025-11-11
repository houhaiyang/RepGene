#!/bin/bash
#DSUB -n lucaOne
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
source /home/share/huadjyin/home/houhaiyang/bashrc/lucaOne.bashrc

# getEmbeddings
cd /home/share/huadjyin/home/houhaiyang/method/lucaOne/LucaOneApp
export CUDA_VISIBLE_DEVICES="0,1,2,3"

data_dir='/home/share/huadjyin/home/houhaiyang/project/RepGene/data/species/Homo_sapiens'

python algorithms/inference_embedding_lucaone.py \
    --llm_dir /home/share/huadjyin/home/houhaiyang/method/lucaOne/models \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10000 \
    --trunc_type right \
    --seq_type gene \
    --input_file ${data_dir}/rna_data.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ${data_dir}/embeddings/LucaOne_RNA \
    --save_type numpy \
    --embedding_type vector \
    --vector_type cls \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 2000 \
    --gpu_id 0

# mergeEmbeddings
cd ${data_dir}
python /home/share/huadjyin/home/houhaiyang/method/lucaOne/LucaOneApp/mergeEmbeddings.py \
    --input_dir ./embeddings/LucaOne_RNA \
    --output_file ./embeddings/LucaOne_RNA_merged_embeddings.csv.gz
