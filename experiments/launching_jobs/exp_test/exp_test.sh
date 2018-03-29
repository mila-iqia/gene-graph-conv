#!/usr/bin/env bash


#python main.py --model cgn --num-layer 2 --num-channel 128 --use-emb 16 --dataset percolate  --lr 0.001 --epoch 200 --cuda --batch-size 100 --extra-cn 10 --pool-graph hierarchy --nb-examples 100

exp_dir=$PWD
cd ../..
./launch_all_my_stuff.sh "--epoch 1 --model cgn --num-layer 2 --num-channel 128 --use-emb 16 --dataset percolate --lr 0.001 --cuda --batch-size 100 --extra-cn 10 --pool-graph hierarchy --nb-examples 100" 3 $exp_dir
./launch_all_my_stuff.sh "--epoch 1 --model cgn --num-layer 3 --num-channel 128 --use-emb 16 --dataset percolate --lr 0.001 --cuda --batch-size 100 --extra-cn 10 --pool-graph hierarchy --nb-examples 100" 2 $exp_dir
