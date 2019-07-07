#!/bin/bash

for i in {1..5}
do
    let "LSTM_NODES=$i*50"
    for k in {1..5}
    do
        let "LAYERS=$k*2"
        for e in {1..3}
        do
            let "ED=$e*100"
            CDFS="files/gs01/fs_$LSTM_NODES""_$LAYERS""_$ED"
            mkdir $CDFS
            python models/train.py --DEVICE=cuda:1 --BATCH_SIZE=500 --CHECKPOINT_DIR=$CDFS --test=few_shot_input/test.txt --train=few_shot_input/train.txt --train=few_shot_input/eng_train.txt --LSTM_NODES=$LSTM_NODES --LEARNING_RATE=0.001 --LAYERS=$LAYERS --SILENT --EMBEDDING_DIM=$ED --EPOCHS=20 > "$CDFS/log.txt"
        done
    done
done
