import numpy as np


def get_description(norm_seq, max_samples=64):
    """
    Return mean, std, min, max of data in dataset.
    :param norm_seq:
    :param max_samples:
    :return:
    """
    train_x = []

    for index, batch in enumerate(norm_seq):
        batch_x, _ = batch
        for x in batch_x:
            train_x.append(x)
        if max_samples is not None and len(train_x) >= max_samples:
            break

    return {'mean': np.mean(train_x), 'std': np.std(train_x), 'min': np.min(train_x), 'max': np.max(train_x)}