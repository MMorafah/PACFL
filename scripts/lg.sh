for trial in 1 2 3
do
    dir='../save_results/lg/noniid-labeldir/cifar100'
    if [ ! -e $dir ]; then
    mkdir -p $dir 
    fi 
    
    python ../main_lg.py --trial=$trial \
    --rounds=200 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=resnet9 \
    --dataset=cifar100 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results/' \
    --partition='noniid-labeldir' \
    --alg='lg' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --cluster_alpha=0.3 \
    --nclasses=10 \
    --nsamples_shared=2500 \
    --gpu=1 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'.txt'

done 
