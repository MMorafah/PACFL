#!/bin/sh

for trial in 3 2 1
do
    dir='../save_results/ifca/homo/mix4'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi 
    
    python ./main_ifca_mix4.py --trial=$trial \
    --rounds=50 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=5 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=mix4 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results/' \
    --partition='homo' \
    --alg='ifca' \
    --nclusters=4 \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'_4clusters.txt'

done 
