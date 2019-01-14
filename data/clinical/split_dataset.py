import os
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from data.clinical import datasets, taskloader

def split_tasks(limit=100, seed=0, normalize=False):

    train_tasks, valid_tasks, test_tasks = [], [], []

    tcga = datasets.TCGADataset()
    tasks = taskloader.get_all_tasks(tcga)
    good_tasks = []

    for task_id in tasks:
        task = datasets.Task(tcga, task_id, limit=limit)
        stats = {}
        for i in task.labels:
            if i in stats:
                stats[i] += 1
            else:
                stats[i] = 1
        #print(task_id)
        #print(stats.values())
        #print(not all(i >= 3 for i in stats.values()))
        if task.get_num_examples() < limit or len(list(set(task.labels))) < 2 or (not all(i >= 10 for i in stats.values())):
            continue
        else:
            good_tasks.append(task)
            #print(task_id)
            #sprint(stats.values())

    #print(len(good_tasks))
    #print(good_tasks)
    num_tasks = len(good_tasks) # 0.7 train 0.2 valid 0.1 test

    train_size = int(num_tasks*0.7)
    valid_size = int(num_tasks*0.2)

    for task in good_tasks[:train_size]:
        train_tasks.append(task)
    for task in good_tasks[train_size:train_size+valid_size]:
        valid_tasks.append(task)
    for task in good_tasks[train_size+valid_size:]:
        test_tasks.append(task)

    #print("Num Train Tasks: " + str(len(train_tasks))) # 185
    #print("Num Valid Tasks: " + str(len(valid_tasks))) # 53
    #print("Num Test Tasks: " + str(len(test_tasks))) # 27
    return train_tasks, valid_tasks, test_tasks


def split_datasets(dataset, batch_size=10, train_size=40, random=True, seed=1993, normalize=False):

    vld_size = 0.3
    tst_size = 0.5
    unique_list = []

    all_idx = range(len(dataset))
    #train_size = train_size
    #valid_size = train_size
    #test_size = 300
    #print(dataset.id)

    for x in dataset.labels:
        if x not in unique_list:
            unique_list.append(x)
    #print(unique_list)

    train_ids, test_ids = train_test_split(all_idx, stratify=dataset.labels,
                                           shuffle=random,
                                           test_size=vld_size,
                                           random_state=seed)
    #print("train:{} " .format(len(train_ids)))

    valid_ids, test_ids = train_test_split(test_ids, stratify=dataset.labels[test_ids],
                                           shuffle=True,
                                           test_size=tst_size,
                                           random_state=seed)
    #print("valid:{} " .format(len(valid_ids)))
    #print("test:{} " .format(len(test_ids)))

    train_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
    valid_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_ids))
    test_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_ids))

    return train_set, valid_set, test_set #67,16,17
