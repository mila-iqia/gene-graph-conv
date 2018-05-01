#!/usr/bin/env bash


#python main.py --model cgn --num-layer 2 --num-channel 128 --use-emb 16 --dataset percolate  --lr 0.001 --epoch 200 --cuda --batch-size 100 --extra-cn 10 --pool-graph hierarchy --nb-examples 100

exp_dir=$PWD
cd ../..

# logistic regression

# With drop out
./launch_all_my_stuff.sh " --model mlp --dataset dgex --graph pancan --nb-examples 2000 --training-mode gene-inference --num-layer 0 --dropout True --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model mlp --dataset dgex --graph pancan --nb-examples 2000 --training-mode gene-inference --num-layer 0 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir

# MLP
# One layer, bigger and bigger.
./launch_all_my_stuff.sh " --model mlp --num-channel 128 --dataset dgex --graph pancan --nb-examples 2000 --training-mode gene-inference --dropout True  --num-layer 1 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model mlp --num-channel 512 --dataset dgex --graph pancan --nb-examples 2000 --training-mode gene-inference --dropout True  --num-layer 1 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model mlp --num-channel 1024 --dataset dgex --graph pancan --nb-examples 2000 --training-mode gene-inference --dropout True  --num-layer 1 --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir

# BIGGER!
./launch_all_my_stuff.sh " --model mlp --num-channel 1024 --num-layer 2 --dataset dgex --graph pancan --nb-examples 2000 --dropout True --training-mode gene-inference --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir
./launch_all_my_stuff.sh " --model mlp --num-channel 1024 --num-layer 3 --dataset dgex --graph pancan --nb-examples 2000 --dropout True --training-mode gene-inference --cuda --data-file bgedv2.hdf5 --data-dir /data/lisa/data/genomics/D-GEX/ --epoch 500 --tensorboard-dir ~/milatmp1/transcriptome/graph/experience/set_1" 1 $exp_dir

# CGN
