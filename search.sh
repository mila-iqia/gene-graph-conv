#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/venv/bin/activate
for i in "$@"
do
case $i in
    -d=*|--search-dataset=*)
    dataset="${i#*=}"

    ;;
    -m=*|--search-model=*)
    model="${i#*=}"
    ;;
esac
done

python -W ignore ~/Documents/code/conv-graph/search.py  --cuda --log=silent --search-model=$model --search-dataset=$dataset
