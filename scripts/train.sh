#!/bin/bash

# train.sh OUTDIR CUDA_DEV

mkdir $1
cp train.sh $1
python models/train.py --DEVICE=cuda:$2 --BATCH_SIZE=1000 --CHECKPOINT_DIR=$1 --val=few_shot_input/test.txt --test=few_shot_input/test.txt --train=few_shot_input/train.txt --train=few_shot_input/eng_train.txt --EPOCHS=60 --LSTM_NODES=300 --LOAD_WEIGHTS="files/fs_06_26_19_01/best_model.pkl" --LEARNING_RATE=0.001
