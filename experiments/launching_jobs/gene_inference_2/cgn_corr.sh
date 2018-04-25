#!/usr/bin/env bash


#python main.py --model cgn --num-layer 2 --num-channel 128 --use-emb 16 --dataset percolate  --lr 0.001 --epoch 200 --cuda --batch-size 100 --extra-cn 10 --pool-graph hierarchy --nb-examples 100

exp_dir=$PWD
cd ../..

# CGN
# Without dropout
./launch_all_my_stuff.sh " --model cgn --dataset dgex --graph train-corr --nb-examples 2000 --training-mode gene-inference --num-layer 1 --use-emb 128 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 250 --num-channel 128 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model cgn --dataset dgex --graph train-corr --nb-examples 2000 --training-mode gene-inference --num-layer 2 --use-emb 128 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 250 --num-channel 128 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model cgn --dataset dgex --graph train-corr --dropout True --nb-examples 2000 --training-mode gene-inference --num-layer 2 --use-emb 128 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --num-channel 128 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model cgn --dataset dgex --graph train-corr --nb-examples 2000 --training-mode gene-inference --num-layer 3 --use-emb 64 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 250 --num-channel 64 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir


