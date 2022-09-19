#!/bin/sh

for trial in 1 2 3
do
    dir='../save_results_unseen/cfl/noniid-#label2/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi 
    
    python ../main_cfl.py --trial=$trial \
    --rounds=50 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=5 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_unseen/' \
    --partition='noniid-#label2' \
    --alg='cfl' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'.txt'

done 
