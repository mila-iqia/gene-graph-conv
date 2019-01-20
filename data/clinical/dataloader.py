from data import split_dataset


def get_dataset(batch_size=40, seed=0):
    # Create the lists of dataset.
    meta_trainset, meta_validset, meta_testset = split_dataset.split_tasks(seed=seed, normalize=False)

    # Split each tasks in train/valid/test set.
    meta_trainset = [split_dataset.split_datasets(d, batch_size=batch_size, seed=seed) for d in meta_trainset]
    meta_validset = [split_dataset.split_datasets(d, batch_size=batch_size, seed=seed) for d in meta_validset]
    meta_testset = [split_dataset.split_datasets(d, batch_size=batch_size, seed=seed) for d in meta_testset]

    return meta_trainset, meta_validset, meta_testset


# Get loaders for sklearn models