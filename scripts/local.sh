for trial in 3
do
    dir='../save_results/local_only/noniid-labeldir/fmnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi 
    
    python ../main_local.py --trial=$trial \
    --rounds=50 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=1 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=fmnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results/' \
    --partition='noniid-labeldir' \
    --alg='local_only' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=1 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'.txt'

done 
