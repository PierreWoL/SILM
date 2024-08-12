#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=2           
#$ -pe smp.pe 2 
#$ -ac nvmps
#$ -m bea
#$ -M zhenyu.wu@manchester.ac.uk

conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
module load mpi/intel-18.0/openmpi/4.0.1-cuda
#module load libs/cuda/11.0.3
module load libs/nccl/2.5.6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"


nvidia-smi
dmesg | grep NVRM
journalctl -k | grep NVRM


nodelist=$(awk '{print $1}' $PE_HOSTFILE | tr '\n' ',' | sed 's/,$//')
echo "Node List: $nodelist"
master_node=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')
echo "Master node: $master_node"
dist_url="tcp://$master_node:40000"
echo "Distributed URL: $dist_url" # #mpirun -np $world_size python -u SwAV.py \ 

num_nodes=$(wc -l < $PE_HOSTFILE)
world_size=2

export TORCH_DISTRIBUTED_DEBUG=DETAIL
#dataset=GDS
datasets=("WDC" "GDS")
#nums=(1000)

for ds in "${datasets[@]}"
  do
    dataPath="datasets/$ds/Test/"
    EXPERIMENT_PATH="./model/swav/$ds/sbert/-1/20/distributed/num500/"
    mkdir -p $EXPERIMENT_PATH
    mpirun -np $world_size python -u SwAV.py \
    --datasetSize -1 \
    --base_lr 0.00005 \
    --data_path $dataPath \
    --nmb_crops 2 0 \
    --hidden_mlp 0 \
    --lm sbert \
    --world_size $world_size \
    --queue_length 0 \
    --crops_for_assign 0 1 \
    --temperature 0.1 \
    --epsilon 0.01 \
    --subject_column \
    --nmb_prototypes 500 \
    --dist_url $dist_url \
    --epochs 150 \
    --batch_size 60 \
    --sinkhorn_iterations 3 \
    --wd 0.000001 \
    --use_fp16 True \
    --dump_path $EXPERIMENT_PATH  
 done
