#!/bin/bash
#DSUB -n Evo2
#DSUB -N 1
#DSUB -A root.project.xxx
#DSUB -q root.default
#DSUB -R "cpu=8;gpu=1;mem=24000"
#DSUB -pn "xxx"
#DSUB -oo /home/share/xxx/home/xxx/project/RepGene/logs/%J.out
#DSUB -eo /home/share/xxx/home/xxx/roject/RepGene/logs/%J.err


# 加载系统 Conda
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
# 加载环境配置
source /home/share/xxx/home/xxx/bashrc/EVO2.bashrc

data_dir='/home/share/xxx/home/xxx/project/RepGene/data/species/Homo_sapiens/'

# getEmbeddings
cd /home/share/xxx/home/xxx/method/Evo2/evo2
# 使用 Evo 生成DNA序列的 Embeddings
python ./test/test_evo2_Embeddings_hou.py \
  --model_name evo2_7b \
   --local_model_path /home/share/xxxhome/xxx/HF_HOME/transformers/arcinstitute/evo2_7b/evo2_7b.pt \
   --layer_name blocks.28.mlp.l3  \
   --embedding_type vector  \
   --vector_type mean  \
   --input_file ${data_dir}/cds_data.csv \
   --output_dir ${data_dir}/embeddings/DNA

# mergeEmbeddings
cd ${data_dir}
python /home/share/xxx/home/xxx/method/Evo2/evo2/test/mergeEmbeddings.py --input_dir ./embeddings/DNA --output_file ./embeddings/DNA_merged_embeddings.csv.gz
