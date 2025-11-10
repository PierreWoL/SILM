#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2           
#$ -pe smp.pe 2 
#$ -ac nvmps
#$ -m bea
#$ -M zhenyu.wu@manchester.ac.uk

conda activate py39
source /mnt/iusers01/fatpou01/compsci01/c29770zw/test/CurrentDataset/datavenv/bin/activate
module load mpi/intel-18.0/openmpi/4.0.1-cuda
module load libs/nccl/2.5.6
echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"


nodelist=$(awk '{print $1}' $PE_HOSTFILE | tr '\n' ',' | sed 's/,$//')
echo "Node List: $nodelist"
master_node=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')
echo "Master node: $master_node"
dist_url="tcp://$master_node:12355"
echo "Distributed URL: $dist_url" # #mpirun -np $world_size python -u SwAV.py \ 
telnet $master_node 12355
num_nodes=$(wc -l < $PE_HOSTFILE)
world_size=2


datasets=("WDC")

for ds in "${datasets[@]}"
  do
    dataPath="datasets/$ds/Test/"
    EXPERIMENT_PATH="./model/swav/$ds/P1/"
    mkdir -p $EXPERIMENT_PATH
    #mpirun -np $world_size python -u SwAV.py \
    torchrun --nproc_per_node=$world_size SwAV.py \
    --datasetSize -1 \
    --base_lr 0.00005 \
    --data_path $dataPath \
    --nmb_crops 1 1 \
    --hidden_mlp 0 \
    --lm sbert \
    --world_size $world_size \
    --queue_length 0 \
    --crops_for_assign 0 1 \
    --temperature 0.1 \
    --epsilon 0.01 \
    --subject_column \
    --column \
    --header \
    --nmb_prototypes 3500 \
    --dist_url $dist_url \
    --epochs 36 \
    --batch_size 60 \
    --cls False \
    --sinkhorn_iterations 3 \
    --wd 0.000001 \
    --dump_path $EXPERIMENT_PATH  
 done
