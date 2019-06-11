#!/bin/bash

# train.sh OUTDIR CUDA_DEV MARGIN

mkdir $1
cp train.sh $1
python models/train.py --DEVICE=cuda:$2 --BATCH_SIZE=1000 --CHECKPOINT_DIR=$1 --test=text2/test.txt --train=text2/train.txt --EPOCHS=60 --LSTM_NODES=100 --LEARNING_RATE 0.001 --MARGIN=$3
