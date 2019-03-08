orion -vv --debug hunt --config bayes.yaml --max-trials 2 --worker-trials 2 ./clinical-tasks-orion.py --lr~'choices([0.1, 0.01, 0.001])' #--agg-reduce~'uniform(1, 10, discrete=True)'
