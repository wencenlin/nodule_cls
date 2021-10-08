#!/bin/bash
set -e


maxeps=4  # 9


for (( i=0; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
    CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 64 --num_epochs 700 --fold $i
done 
