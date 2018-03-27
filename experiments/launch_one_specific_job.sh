#!/usr/bin/env bash


# Launch one thing. Depends on local stuff. change this file if needed.
source ~/venv/bin/activate
command=$@
#python -W ignore ../../../baseline.py --log=silent --cuda --baseline-model=$model --baseline-dataset=$dataset

echo "Launching python main.py $command"
cd ..
python -u main.py $command --tensorboard-dir ./experiments/experiments/
