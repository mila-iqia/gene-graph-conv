import pandas as pd

DIR = 'data/clinical/'

def get_all_tasks(tcga, seed = 0):
    columns = pd.read_csv(DIR + 'tcga_cols.csv', header=None, index_col=0)
    task_ids = []

    for index, col_name in columns.iterrows():
        labels = tcga.labels.loc(axis=1)[col_name[1], 'clinicalMatrix'].dropna()
        #print("\n number of tcga labels for this attribute: {}" .format(len(labels)))
        matrices = labels['clinicalMatrix'].unique()
        #print("number of tissue type for this attribute: {}" .format(len(matrices)))
        temp_task_ids = []

        for matrix in matrices:
            task = labels[labels['clinicalMatrix'] == matrix]
            task_id = col_name[1] + "-" + matrix
            if len(task[col_name[1]].unique()) > 1 and len(task[col_name[1]].unique()) < 10:
                temp_task_ids.append(task_id)
                #print("\n\n ****** attribute, tissue: {}, {}" .format(col_name[1], matrix))
                #print("\nunique labels for this attribute: {} " .format(task[col_name[1]].unique()))
                #print("\nnumber of unique labels for this attribute:{}" .format(len(task[col_name[1]].unique())))
        if len(temp_task_ids) != 1:
            task_ids = task_ids + temp_task_ids
    return task_ids

# 345 tasks
