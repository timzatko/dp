import os
import numpy as np


def sequence_to_numpy(seq, path, name, input_shape=None):
    """
    Converts sequence to numpy array. If the sequence exists in storage, load it.
    """
    path_x = os.path.join(path, f'{name}_x.npy')
    path_y = os.path.join(path, f'{name}_y.npy')

    if input_shape is None:
        input_shape = seq.input_shape

    if os.path.exists(path_x):
        print(f'loading {path_x}, {path_y}...')

        with open(path_x, 'rb') as f:
            train_x = np.load(f, allow_pickle=True)
        with open(path_y, 'rb') as f:
            train_y = np.load(f, allow_pickle=True)

        return train_x.reshape((-1, *input_shape)), __train_y_reshape(seq, train_y)

    print(f'generating {path_x}, {path_y}...')

    train_x = []
    train_y = []

    for batch_x, batch_y, *r in seq:
        if batch_x is not None and batch_y is not None:
            for x, y in zip(batch_x, batch_y):
                train_x.append(x)
                train_y.append(y)

    with open(path_x, 'wb') as f:
        np.save(f, np.array(train_x))
    with open(path_y, 'wb') as f:
        np.save(f, np.array(train_y))

    return np.array(train_x).reshape((-1, *input_shape)), __train_y_reshape(seq, train_y)


def __train_y_reshape(seq, train_y):
    if not seq.one_hot:
        return np.array(train_y).reshape(-1)
    return np.array(train_y).reshape(-1, len(seq.class_names))