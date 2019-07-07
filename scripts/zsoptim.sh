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
            CDZS="files/gs01/zs_$LSTM_NODES""_$LAYERS""_$ED"
            mkdir $CDZS
            python models/train.py --DEVICE=cuda:0 --BATCH_SIZE=500 --CHECKPOINT_DIR=$CDZS --val=zero_shot_input/val.txt --test=zero_shot_input/test.txt --train=zero_shot_input/train.txt --train=zero_shot_input/eng_train.txt --LSTM_NODES=$LSTM_NODES --LEARNING_RATE=0.001 --EMBEDDING_DIM=$ED --SILENT --LAYERS=$LAYERS --EPOCHS=20 > "$CDZS/log.txt"
        done
    done
done
