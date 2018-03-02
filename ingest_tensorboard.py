import pickle
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



# walk the experiment directory
experiments = {}
for root, dirs, files in os.walk('experiments'):
    experiment = defaultdict(dict)
    for f in files:
        key = '/'.join(root.split('/')[1:-1])
        if f == "options.pkl":
            opts = vars(pickle.load(open(os.path.join(root, f), 'rb')))
            trial = opts['trial_number']
            experiment = experiments.get(key, defaultdict(dict))
            experiment[trial]['opts'] = opts
            experiments[key] = experiment
        elif f.startswith("event"):
            trial = root.split('/')[-1]
            experiment = experiments.get(key, defaultdict(dict))
            experiment[trial]['event_accumulator'] = EventAccumulator(os.path.join(root, f))
            experiments[key] = experiment

trial = experiments.values()[0].values()[0]
scalar_tags = trial['event_accumulator'].Reload().Tags()['scalars']
opt_tags = trial['opts'].keys()
df = pd.DataFrame(columns=opt_tags + scalar_tags)
exp_df = pd.DataFrame(columns=opt_tags + scalar_tags)
for path, experiment in experiments.iteritems():
    for number, trial in experiment.iteritems():
        event_accumulator = trial['event_accumulator'].Reload()
        for tag in scalar_tags:
            events = event_accumulator.Scalars(tag)
            scalars = np.array(map(lambda x: x.value, events))
            df.loc[:, tag] = scalars
        for tag in opt_tags:
            df.loc[:, tag] = [trial['opts'][tag]]
        exp_df = exp_df.append(df)
print exp_df
import pdb; pdb.set_trace()

for path, experiment in experiments.iteritems():
    for number, trial in experiment.iteritems():
        print trial
event_acc.Reload()

# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('accuracy_test'))

# Get the accuracy from the first epochs
event_acc.Scalars('accuracy_test')[0][2]
