import math
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras.utils import Sequence


def my_cmp(v):
    return v[0]


# [heat, voxel, (z, y, x)]
def value_to_index(heatmap, image_x, sort='ASC'):
    values = []

    for z, _ in enumerate(heatmap):
        for y, _ in enumerate(heatmap[z]):
            for x, _ in enumerate(heatmap[z][y]):
                heat = heatmap[z][y][x]
                voxel = image_x[z][y][x]
                values.append((heat, voxel, (z, y, x)))

    reverse = sort == 'DESC'
    values.sort(reverse=reverse, key=my_cmp)

    return np.array(values)


class EvaluationSequence(Sequence):
    def __init__(self, t, image, heatmap, step_size=1, max_steps=1000, batch_size=8, debug=False, log=True):
        self.t = t
        self.log = log
        self.debug = debug
        self.batch_size = batch_size
        self.image = image
        self.heatmap = heatmap

        # cache the images if debug is enabled
        self.cache = []

        # this is insertion
        # [heat, voxel, (z, y, x)]
        self.voxels = value_to_index(heatmap, image, sort='DESC')

        if self.t == 'insertion':
            self.new_image = np.zeros(shape=image.shape)
        else:
            self.new_image = np.copy(image)

        self.step_size = step_size
        self.steps = 0
        self.max_steps = self.__get_max_steps(max_steps)
        self.max_voxels = self.max_steps * step_size

        if self.log:
            print(f'max_steps: {self.max_steps}, batch_size: {self.batch_size}')

        self.tqdm = None
        if self.log:
            self.tqdm = tqdm(total=self.__len__())

    def __len__(self):
        return math.ceil(self.max_steps / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []
        if self.tqdm is not None:
            self.tqdm.update()

        for i in range(self.batch_size):
            step = idx * self.batch_size + i
            # print(f'step: {step}')

            if step >= self.max_steps:
                break;

            start = step * self.step_size
            end = start + self.step_size
            # print(f'{start},{end}')

            # select voxels to add to the image
            s_voxels = self.voxels[start:end]

            for _, voxel, index in s_voxels:
                z, y, x = index

                if self.t == 'insertion':
                    # set the value to the voxel (on mutable image)
                    self.new_image[z][y][x] = voxel
                else:
                    # deletion, remove the voxel
                    self.new_image[z][y][x] = 0

            # append a copy, because we will mutate the image in next step
            batch_x.append(np.copy(self.new_image))

            if self.debug:
                self.cache.append(np.copy(self.new_image))

        batch_x = np.array(batch_x).reshape(-1, *self.new_image.shape)

        return batch_x

    def __get_max_steps(self, max_steps):
        return min(math.ceil(len(self.voxels) / self.step_size), max_steps)


def get_curve(image_y, y_pred, step_size):
    idx = image_y.argmax(axis=0)
    y = y_pred[:, idx]
    x = np.array(list(map(lambda s: s * step_size, range(len(y)))))
    return x, y


def evaluation_auc(image_y, y_pred, step_size):
    x, y = get_curve(image_y, y_pred, step_size)
    return sklearn.metrics.auc(x, y)


def plot_evaluation(image_y, y_pred, eval_seq, title='insertion'):
    idx = image_y.argmax(axis=0)
    x, y = get_curve(image_y, y_pred, eval_seq.step_size)
    auc = evaluation_auc(image_y, y_pred, eval_seq.step_size)
    plt.title(f'{title}: auc={auc}, y_true={idx}, voxel_count:{eval_seq.max_voxels:,} / {len(eval_seq.voxels):,})')
    plt.plot(x, y, linewidth=2)
    ax = plt.gca()
    ax.set_ylabel(f'activation')
    ax.set_xlabel(f'voxels')


def predict_seq_as_np(model, eval_seq, batch_size, log=False):
    """
    predict_seq_as_np
    eval_seq = sequence to evaluate
    batch_size = model.predict() batch size
    """
    y_pred = None
    count = len(eval_seq)

    for i, batch_x in enumerate(eval_seq):
        if log:
            print(f'evaluating batch {i}/{count} of length {len(batch_x)}...')
        batch_y_pred = model.predict(batch_x, batch_size=batch_size)
        if y_pred is None:
            y_pred = batch_y_pred
        else:
            y_pred = np.concatenate([y_pred, batch_y_pred], axis=0)

    return y_pred
