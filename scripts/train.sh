#!/bin/bash

# train.sh OUTDIR CUDA_DEV

mkdir $1
cp scripts/train.sh $1
python models/train.py --DEVICE=cuda:$2 --BATCH_SIZE=1000 --CHECKPOINT_DIR=$1 --input="../data/deduped/all_case/few_shot" --minput="../data/eng_train.txt" --HIDDEN=400 --WORD_EMBEDDING=200 --CHAR_EMBEDDING=200 --LOAD_WEIGHTS="garbage/best_model.pkl"
