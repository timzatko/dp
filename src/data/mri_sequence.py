import os
import math

import numpy as np
import SimpleITK as sitk

from tensorflow.keras.utils import Sequence

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from skimage.transform import resize


def process_image(path, input_shape, resize_img, normalization):
    x = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if resize_img:
        x = resize(x, input_shape[:3])
    if normalization is not None:
        x = normalize(x, normalization)
    return np.array(x).reshape(input_shape)


def invert_img(x):
    # [x, y, z, 1]
    return x[:, :, ::-1, :]


def normalize(x, normalization):
    desc = normalization['desc']
    if normalization['type'] != 'standardization':
        return (x - desc['min']) / (desc['max'] - desc['min'])
    return (x - desc['mean']) / desc['std']


def apply_augmentation(x, augmentations, name, force=False):
    if name == 'random_swap_hemispheres':
        p = augmentations['random_swap_hemispheres']
        if force or np.random.uniform(0) < p:
            return invert_img(x), True
    return x, False


def apply_augmentations(x, augmentations):
    if augmentations is None:
        return x

    for augmentation in augmentations.keys():
        x, applied = apply_augmentation(x, augmentations, augmentation)
        if applied:
            return x

    return x


def readfile(file_path):
    fo = open(file_path, "r")
    c = fo.readline()
    fo.close()
    return c


# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class MRISequence(Sequence):
    def __init__(self, path, batch_size, input_shape, class_names=['AD', 'CN'],
                 augmentations=None, augmentations_inplace=True, images=True, one_hot=True, class_weight=None,
                 normalization=None, resize_img=True):
        if one_hot == False:
            self.encoder = LabelEncoder()
            self.encoder.fit(np.array(class_names))
        else:
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.fit(np.array(class_names).reshape(-1, 1))

        self.class_weight = class_weight
        self.one_hot = one_hot
        self.input_shape = input_shape
        self.resize_img = resize_img
        self.class_names = class_names
        self.images = images
        self.augmentations = augmentations
        self.augmentations_inplace = augmentations_inplace
        self.normalization = normalization

        self.batch_size = batch_size
        self.images_dirs = [os.path.join(path, key) for key in os.listdir(path)]

    def __len__(self):
        return math.ceil(len(self.images_dirs) / self.batch_size)
        # Uncomment for debugging (when you need a smaller subset of data and faster training time)
        # return math.ceil(18 / self.batch_size)

    def __getitem__(self, idx):
        images_dirs = self.images_dirs[idx * self.batch_size:(idx + 1) * self.batch_size]

        if not len(images_dirs):
            batch_y = np.array([]).reshape(-1)
            if self.one_hot:
                batch_y = np.array([]).reshape(-1, len(self.class_names))

            return np.array([]).reshape(-1, *self.input_shape), batch_y

        batch_y = self.__encode(
            np.array([readfile(os.path.join(image_dir, 'real_diagnosis.txt')) for image_dir in images_dirs]))

        # if we disabled loading images, don't do it
        if not self.images:
            batch_x = np.array([None for image_dir in images_dirs])
        else:
            batch_x = np.array([process_image(os.path.join(image_dir, 'data.nii'), self.input_shape, self.resize_img,
                                              self.normalization) for image_dir in images_dirs])

            if self.augmentations:
                if self.augmentations_inplace:
                    batch_x = np.array([apply_augmentations(x, self.augmentations) for x in batch_x])
                else:
                    new_batch_x = np.array([]).reshape(-1, *self.input_shape)
                    new_batch_y = np.array([]).reshape(-1, len(self.class_names))

                    for augmentation in self.augmentations.keys():
                        aug_batch_x = np.array([x for x, _ in
                                                [apply_augmentation(x, self.augmentations, augmentation, force=True) for x in
                                                 batch_x]])
                        new_batch_x = np.concatenate((new_batch_x, aug_batch_x), axis=0)
                        new_batch_y = np.concatenate((new_batch_y, np.copy(batch_y)), axis=0)

                    batch_x = np.concatenate((batch_x, new_batch_x), axis=0)
                    batch_y = np.concatenate((batch_y, new_batch_y), axis=0)

        if self.class_weight is None:
            return batch_x, batch_y

        batch_w = np.array([self.class_weight[y] for y in self.__decode(batch_y)])
        return batch_x, batch_y, batch_w

    def __encode(self, labels):
        if self.one_hot:
            labels = labels.reshape(-1, 1)
        return self.encoder.transform(labels)

    def __decode(self, labels):
        if self.one_hot:
            return np.argmax(labels, axis=1)
        return labels
