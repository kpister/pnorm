#!/bin/bash

# train.sh OUTDIR CUDA_DEV MARGIN

mkdir $1
cp train.sh $1
python models/train.py --DEVICE=cuda:$2 --BATCH_SIZE=500 --CHECKPOINT_DIR=$1 --val=zero_shot_input/val.txt --test=zero_shot_input/test.txt --train=zero_shot_input/train.txt --train=zero_shot_input/eng_train.txt --EPOCHS=60 --LSTM_NODES=100 --LEARNING_RATE 0.001 --MARGIN=$3
