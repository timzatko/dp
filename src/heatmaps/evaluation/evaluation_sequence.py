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
    assert heatmap.shape == image_x.shape

    for z in range(len(heatmap)):
        for y in range(len(heatmap[z])):
            for x in range(len(heatmap[z][y])):
                heat = heatmap[z][y][x]
                voxel = image_x[z][y][x]
                values.append((heat, voxel, (z, y, x)))
      
    reverse = sort == 'DESC'
    values.sort(reverse=reverse, key=lambda v: v[0])
    
    return values


class EvaluationSequence(Sequence):
    def __init__(self, t, image, heatmap, step_size=1, max_steps=1000, batch_size=8, default=0, debug=False, log=True):
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
        self.default = default

        # cache the images if debug is enabled
        self.cache = []

        # this is insertion
        # [heat, voxel, (z, y, x)]
        self.voxels = value_to_index(heatmap, image, sort='DESC')

        if self.t == 'insertion':
            # ones vs zeros, what is better?
            if default:
                self.new_image = np.ones(shape=image.shape)
            else:
                self.new_image = np.zeros(shape=image.shape)
        else:
            self.new_image = np.copy(image)

        self.step_size = step_size
        self.steps = 0
        self.max_steps = self.__get_max_steps(max_steps)
        self.max_voxels = self.new_image.shape[0] * self.new_image.shape[1] * self.new_image.shape[2]

        if self.log:
            print(f'max_steps: {self.max_steps}, batch_size: {self.batch_size}')

        self.progress_bar = None
        if self.log:
            self.progress_bar = tqdm(total=self.__len__())

    def __len__(self):
        return math.ceil(self.max_steps / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []
        if self.progress_bar is not None:
            self.progress_bar.update()

        for i in range(self.batch_size):
            step = idx * self.batch_size + i
            # print(f'step: {step}')

            if self.max_steps < step:
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
                    self.new_image[z][y][x] = self.default

            # append a copy, because we will mutate the image in next step
            batch_x.append(np.copy(self.new_image))

            if self.debug:
                self.cache.append(np.copy(self.new_image))

        batch_x = np.array(batch_x).reshape(-1, *self.new_image.shape)

        return batch_x

    def __get_max_steps(self, max_steps):
        steps = math.ceil(len(self.voxels) / self.step_size)
        return min(steps, max_steps) if max_steps >= 0 else steps
