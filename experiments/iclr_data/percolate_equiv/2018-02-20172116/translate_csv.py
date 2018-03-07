import os
import pandas as pd
files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(f) and 'slurm' in f]
df = pd.DataFrame()
for f in sorted(files):
    if os.stat(f).st_size != 0:
        df_temp = pd.read_csv(f)
        df = df.append(df_temp.round(3))
df.test_auc_std = df.test_auc_std * 100
df.test_auc_std = df.test_auc_std.apply( lambda x : str(x) + '%')
df.test_auc_mean = df.test_auc_mean * 100
df.test_auc_mean = df.test_auc_mean.apply( lambda x : str(x) + '%')
print df[['model', 'test_auc_mean', 'test_auc_std']].to_string(index=False)
