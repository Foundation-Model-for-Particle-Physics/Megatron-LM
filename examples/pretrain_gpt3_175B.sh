#!/bin/bash
#SBATCH -A m3443
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -L scratch,cfs
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --time=0:30:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -N 256
#SBATCH -J odd_gpt3




DIR="/global/homes/x/xju/scratch/LLMTracking/Megatron-LM"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b


DATASET_1="data/odd-v2-padded-indexed"
DATASET_2="data/odd-v3-padded-indexed"
# DATASET="0.5 ${DATASET_1} 0.5 ${DATASET_2}"
DATASET=$DATASET_2

CHECKPOINT_PATH=run/gpt3-175B-odd-padded
TB_PATH=${CHECKPOINT_PATH}/tensorboard
VOCAB_FILE=configs/odd-vocab.txt

options=" \
	--tensor-model-parallel-size 4 \
	--pipeline-model-parallel-size 16 \
        --num-layers 96 \
        --hidden-size 12288 \
        --num-attention-heads 96 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size 1 \
	--global-batch-size 1536 \
	--rampup-batch-size 16 16 5859375 \
	--train-samples 146484375 \
       	--lr-decay-samples 126953125 \
        --lr-warmup-samples 183105 \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 40 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file $VOCAB_FILE \
	--save-interval 100 \
	--save $CHECKPOINT_PATH \
	--load $CHECKPOINT_PATH \
	--split 98,1,1 \
	--clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
	--tensorboard-dir $TB_PATH \
	--attention-softmax-in-fp32 \
	--fp16 "


# run_cmd="python -u ${DIR}/pretrain_gpt.py $@ ${options}"


# srun -l \
#      --container-image "nvcr.io/nvidia/pytorch:20.12-py3" \
#      --container-mounts "<DIRECTORIES TO MOUNT>" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"


run_cmd="python -u pretrain_gpt.py $@ ${options}"

podman-hpc run -it --shm-size=2g --rm --gpu \
  -v /pscratch/sd/x/xju/LLMTracking/Megatron-LM:/workspace/Megatron-LM \
  -w /workspace/Megatron-LM \
  -v /global/cfs/cdirs/m3443/data/ODD:/workspace/data nvcr.io/nvidia/pytorch:24.03-py3 bash \
  -c "${run_cmd}"

set +x
