import tensorflow as tf
import numpy as np

from skimage import filters
from skimage.util import random_noise


def augment_invert_img(x):
    return x[:, :, ::-1, :]


def augment_rotate_img(x, angle=None):
    x_rotated = tf.keras.preprocessing.image.random_rotation(
        x.reshape(x.shape[:3]),
        angle,
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode='nearest',
        cval=0.0,
        interpolation_order=1
    )
    return x_rotated.reshape((*x_rotated.shape, 1))


def augment_random_zoom(x, zoom=None):
    x_zoomed = tf.keras.preprocessing.image.random_zoom(
        x.reshape(x.shape[:3]),
        (1 - zoom, zoom),
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode='constant',
        cval=0.0,
        interpolation_order=1
    )
    return x_zoomed.reshape((*x_zoomed.shape, 1))


def augment_random_shear(x, angle=None):
    x_shared = tf.keras.preprocessing.image.random_shear(
        x.reshape(x.shape[:3]), angle, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest',
        cval=0.0, interpolation_order=1
    )
    return x_shared.reshape((*x_shared.shape, 1))


def augment_random_gaussian_blur(x, sigma=None):
    # TODO: fix this, sigma should be also random from an sinterval
    x = np.array(x.reshape(x.shape[:3]))
    seed = None if 'RANDOM_SEED' not in globals() else globals()['RANDOM_SEED']
    sigma = sigma + tf.random.uniform([3], 0, 1) * (1 - sigma)
    return filters.gaussian(x, multichannel=True, sigma=sigma).reshape((*x.shape, 1))


def augment_random_gaussian_noise(x, var=None):
    x = np.array(x.reshape(x.shape[:3]))
    seed = None if 'RANDOM_SEED' not in globals() else globals()['RANDOM_SEED']
    var = tf.random.uniform([], 0, 1) * var
    x = random_noise(x, mode='gaussian', seed=seed, var=var, clip=True)
    return x.reshape((*x.shape, 1))


def get_augment_fn(options):
    if options is None:
        options = {
            'invert': (0.5, None),
            'rotate': (0.2, 5),
            'zoom': (0.2, 0.015),
            'shear': (0.2, 2.5),
            'blur': (0.2, 0.8),
            'noise': (0.2, 0.00025)
        }
   
    def augment(x): 
        x = x.numpy()

        if options['invert'][0] and tf.random.uniform([], 0, 1) < options['invert'][0]:
            x = augment_invert_img(x)

        if options['rotate'][0] and tf.random.uniform([], 0, 1) < options['rotate'][0]:
            x = augment_rotate_img(x, angle=options['rotate'][1])

        if options['zoom'][0] and tf.random.uniform([], 0, 1) < options['zoom'][0]:
            x = augment_random_zoom(x, zoom=options['zoom'][1])

        if options['shear'][0] and tf.random.uniform([], 0, 1) < options['shear'][0]:
            x = augment_random_shear(x, options['shear'][1])

        if options['blur'][0] > 0 and tf.random.uniform([], 0, 1) < options['blur'][0]:
            x = augment_random_gaussian_blur(x, options['blur'][1])

        if options['noise'][0] > 0 and tf.random.uniform([], 0, 1) < options['noise'][0]:
            x = augment_random_gaussian_noise(x, var=options['noise'][1])
        
        return x

    return augment


def get_class_weight(y):
    class_weights = {0: 0.8072289156626505, 1: 1.3137254901960784}
    return class_weights[int(np.argmax(y.numpy(), axis=0))]
