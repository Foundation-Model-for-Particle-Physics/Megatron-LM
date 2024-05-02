#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=run/gpt2-345m-odd-v2-padded-indexed
VOCAB_FILE=configs/odd-vocab.txt
# MERGE_FILE=<Path to merges.txt (e.g. /gpt2-merges.txt)>

export CUDA_DEVICE_MAX_CONNECTIONS=1

# pip install flask-restful

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 24  \
       --hidden-size 1024  \
       --load ${CHECKPOINT}  \
       --num-attention-heads 16  \
       --max-position-embeddings 1024  \
       --tokenizer-type GPT2ModuleIDTokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --vocab-file $VOCAB_FILE  \
       --seed 42 \
       --attention-softmax-in-fp32
