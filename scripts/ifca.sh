for trial in 1
do
    dir='../save_results/ifca/noniid-#label1/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi 
    
    python ../main_ifca2.py --trial=$trial \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=1 \
    --local_bs=20 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results/' \
    --partition='noniid-#label1' \
    --alg='ifca' \
    --beta=0.1 \
    --local_view \
    --nclusters=2 \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'.txt'

done 
