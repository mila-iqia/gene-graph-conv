orion -vv hunt --config bayes.yaml --branch clinical15 --max-trial=100 --code-change-type noeffect ./clinical-tasks-orion.py --lr~'loguniform(0.0001, 0.01)' #--agg-reduce~'loguniform(1., 10.)'
