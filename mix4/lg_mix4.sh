#!/bin/sh

for trial in 51 52 53
do
    dir='../save_results/lg/homo/mix4'
    if [ ! -e $dir ]; then
    mkdir -p $dir 
    fi 
    
    python ./main_lg_mix4.py --trial=$trial \
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
    --alg='lg' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --cluster_alpha=0.3 \
    --nclasses=10 \
    --nsamples_shared=2500 \
    --gpu=0 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'.txt'

done 
