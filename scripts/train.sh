#!/bin/bash

# train.sh OUTDIR CUDA_DEV

mkdir $1
cp scripts/train.sh $1
python models/train.py --DEVICE=cuda:$2 --BATCH_SIZE=1000 --CHECKPOINT_DIR=$1 --input="../data/deduped/both/few_shot" --minput="../data/eng_train.txt" 
