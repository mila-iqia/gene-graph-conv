#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/venv/bin/activate
for i in "$@"
do
case $i in
    -d=*|--baseline-dataset=*)
    dataset="${i#*=}"

    ;;
    -m=*|--baseline-model=*)
    model="${i#*=}"
    ;;
esac
done
python -W ignore ~/Documents/code/conv-graph/baseline.py  --cuda --log=silent --baseline-model=$model --baseline-dataset=$dataset
