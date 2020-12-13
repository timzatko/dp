import math
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def get_heatmap(x, y, model, risei, batch_size=8, masks_count=24, risei_batch_size=480, debug=False, log=True):
    """
    batch_size - model batch size
    """
    cls_idx = np.argmax(y)

    batch_count = math.ceil(masks_count / risei_batch_size)

    heatmap = np.zeros(shape=x.shape[:3])
    weights = 0

    masks_x = [] if debug else None
    masks_y = [] if debug else None

    for batch_idx in range(batch_count):
        if debug:
            print(f'\nbatch #{batch_idx} of {batch_count}')

        batch_masks_count = min(risei_batch_size, masks_count - batch_idx * risei_batch_size)
        # Reshape input to risei without channels,
        # then reshape masks back with channels.
        batch_x, masks = risei.generate_masks(batch_masks_count, x.reshape(x.shape[:3]), log=log)
        batch_y = [y for _ in range(batch_masks_count)]

        y_pred_per_mask = model.predict(batch_x.reshape((-1, *x.shape)), batch_size=batch_size)

        for mask, mask_x, y_pred in zip(masks, batch_x, y_pred_per_mask):
            # invert the mask, since 1 is for no masking
            # y is the activation for that class on the last layer
            heatmap = heatmap + y_pred[cls_idx] * (1 - mask)
            weights += y_pred[cls_idx]

            if debug:
                masks_x.append(mask_x)
                masks_y.append(y_pred[cls_idx])

    if debug:
        print(f'\n\ny_true: {cls_idx}')

    heatmap = heatmap / weights

    return heatmap, np.array(masks_x), np.array(masks_y)


def to_gray_scale(img):
    return (img * 255).astype(np.uint8)


def img_norm(image_x):
    return (image_x - image_x.min()) / (image_x.max() - image_x.min())


def show_heatmap(image_x, heatmap, z=None, alpha=0.5):
    """
    image_x - source image of shape (z, x, y, 1)
    heatmaps - a generated heatmaps of shape (z, x, y)
    """
    if z is None:
        z = math.ceil(image_x.shape[0] / 2)

    # we need to convert a colormap because 0 is red and 1 is blue
    heatmap_grayscale = to_gray_scale(1 - heatmap[z])
    heatmap_cmap = cv2.applyColorMap(heatmap_grayscale, cv2.COLORMAP_JET)

    image_x_grayscale = to_gray_scale(img_norm(image_x[z].reshape(image_x[z].shape[:2])))
    image_x_cmap = cv2.applyColorMap(image_x_grayscale, cv2.COLORMAP_BONE)

    return Image.blend(Image.fromarray(image_x_cmap, mode='RGB'), Image.fromarray(heatmap_cmap, mode='RGB'), alpha)


def show_mask(index, masks_x, masks_y, z):
    mask = masks_x[index]
    plt.title(f'y_pred:{masks_y[index]}')
    plt.imshow(mask[z].reshape(mask.shape[1:3]))


def plot_heatmap(image_x, image_y, y_pred, heatmap):
    idx = np.argmax(image_y, axis=0)

    plt.subplot(1, 2, 1)
    plt.title(f'y_true: {idx}, y_pred: {y_pred}')
    plt.imshow(image_x.reshape(image_x.shape[:2]))
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.title(f'y_true: {idx}, y_pred: {y_pred}')
    plt.imshow(image_x.reshape(image_x.shape[:2]))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()


def plot_heatmap_z(image_x, image_y, y_pred, heatmap, z):
    plot_heatmap(image_x[z, :, :], image_y, y_pred, heatmap[z, :, :])


def plot_heatmap_y(image_x, image_y, y_pred, heatmap, y):
    plot_heatmap(image_x[:, y, :], image_y, y_pred, heatmap[:, y, :])


def plot_heatmap_x(image_x, image_y, y_pred, heatmap, x):
    plot_heatmap(image_x[:, :, x], image_y, y_pred, heatmap[:, :, x])