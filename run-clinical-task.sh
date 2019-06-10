#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc

for i in {0..19}; do
    python3 -u clinical-task-hpsearch.py -seed $i $@
done


