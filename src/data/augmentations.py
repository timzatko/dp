import tensorflow as tf
import numpy as np

from skimage import filters


def augment_invert_img(x):
    return x[:, :, ::-1, :]


def augment_rotate_img(x, angle=15.):
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


def augment_random_zoom(x, zoom=0.1):
    x_zoomed = tf.keras.preprocessing.image.random_zoom(
        x.reshape(x.shape[:3]),
        (1 - zoom, 1 - zoom),
        row_axis=1,
        col_axis=2,
        channel_axis=0,
        fill_mode='nearest',
        cval=0.0,
        interpolation_order=1
    )
    return x_zoomed.reshape((*x_zoomed.shape, 1))


def augment_random_shear(x, angle=90.):
    x_shared = tf.keras.preprocessing.image.random_shear(
        x.reshape(x.shape[:3]), angle, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest',
        cval=0.0, interpolation_order=1
    )
    return x_shared.reshape((*x_shared.shape, 1))


def augment_random_gaussian_blur(x, sigma=1.):
    x = np.array(x.reshape(x.shape[:3]))
    return filters.gaussian(x, multichannel=True, sigma=sigma).reshape((*x.shape, 1))


def augment(x):
    x = x.numpy()

    if tf.random.uniform([], 0, 1) < 0.5:
        x = augment_invert_img(x)

    if tf.random.uniform([], 0, 1) < 0.15:
        x = augment_rotate_img(x, angle=10)

    if tf.random.uniform([], 0, 1) < 0.15:
        x = augment_random_zoom(x, zoom=0.01)

    if tf.random.uniform([], 0, 1) < 0.15:
        x = augment_random_shear(x, angle=2.5)

    if tf.random.uniform([], 0, 1) < 0.15:
        x = augment_random_gaussian_blur(x, sigma=0.8)

    return x


def get_class_weight(y):
    class_weight = {0: 1.2641509433962266, 1: 0.8271604938271606}
    return class_weight[int(np.argmax(y.numpy(), axis=0))]
