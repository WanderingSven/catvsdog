#!/bin/bash

stage=1

traing_set=../data/microsoft-catvsdog/PetImages
dev_set=../data/microsoft-catvsdog/dev
test_set=../data/microsoft-catvsdog/test
ngpu=4
tag=vgg
out_path=exp/$tag/

mkdir -p $out_path

if [ $stage -le 1 ]; then
    echo "1.training>>>>>>>>>>>>>>>"
    touch $out_path/train.log
    python  -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 training.py -c $traing_set  -n 2 -o $out_path
    
fi

if [ $stage -le 2 ]; then
    echo "2.testing>>>>>>>>>>>>>>>>"
    python3 testing.py --test_dataset $test_set -n 0 -s $out_path/vgg.ep32
fi
