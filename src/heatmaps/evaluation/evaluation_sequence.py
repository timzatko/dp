import math

import numpy as np

from tqdm import tqdm
from tensorflow.keras.utils import Sequence


def value_to_index(heatmap, image_x, sort='ASC'):
    """
    Reindex heatmap and image to: [heat/activation, voxel, (z, y, x)]
    :param heatmap:
    :param image_x:
    :param sort:
    :return:
    """
    values = []

    for z, _ in enumerate(heatmap):
        for y, _ in enumerate(heatmap[z]):
            for x, _ in enumerate(heatmap[z][y]):
                heat = heatmap[z][y][x]
                voxel = image_x[z][y][x]
                values.append((heat, voxel, (z, y, x)))

    reverse = sort == 'DESC'
    values.sort(reverse=reverse, key=lambda v: v[0])

    return np.array(values)


class EvaluationSequence(Sequence):
    def __init__(self, t, image, heatmap, step_size=1, max_steps=1000, batch_size=8, debug=False, log=True):
        """
        Create a sequence of images from original image and heatmap, by removing/inserting most
        important pixels from it.
        :param t: type - insertion or deletion
        :param image: original image
        :param heatmap: calculated heatmap for that image
        :param step_size: how many voxels are deleted/inserted in one step - image
        :param max_steps: maximum number of steps - generated images
        :param batch_size: how many images return in one batch
        :param debug: enable debug mode
        :param log: enable logs
        """
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

        self.bar = None
        if self.log:
            self.bar = tqdm(total=self.__len__())

    def __len__(self):
        return math.ceil(self.max_steps / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []
        if self.bar is not None:
            self.bar.update()

        for i in range(self.batch_size):
            step = idx * self.batch_size + i
            # print(f'step: {step}')

            if 0 < self.max_steps <= step:
                break

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


