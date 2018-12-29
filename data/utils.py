import os
import pickle 


def record_result(results, experiment, filename):
    results = results.append(experiment, ignore_index=True)
    results_dir = "/".join(filename.split('/')[0:-1])

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    pickle.dump(results, open(filename, "wb"))
    return results

def get_file_separator(filename):
    separators = {'.tsv' : '\t', '.txt': '\t', '.csv': ','}
    sep = separators[os.path.splitext(filename.replace('.gz', ''))[-1]]
    return sep