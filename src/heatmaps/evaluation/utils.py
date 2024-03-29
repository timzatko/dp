import numpy as np

from sklearn import metrics
from matplotlib import pyplot as plt


def get_curve(image_y, y_pred, step_size):
    """
    Calculate curve, x a y for activations.
    """
    idx = image_y.argmax(axis=0)
    y = y_pred[:, idx]
    x = np.array(list(map(lambda s: s * step_size, range(len(y)))))
    return x, y


def evaluation_auc(image_y, y_pred, step_size):
    """
    Get area under curve from activation over time (inserted/deleted pixels).
    """
    x, y = get_curve(image_y, y_pred, step_size)
    return metrics.auc(x, y)


def plot_evaluation(image_y, y_pred, step_size, voxels, max_voxels, title='insertion'):
    """
    Plot evaluation graph.
    :param max_voxels: number of voxels evaluated (inserted/deleted) voxels
    :param voxels: number of voxels in the image
    :param step_size: step size of evaluation sequence
    :param image_y: true y
    :param y_pred: predictions from evaluation sequence
    :param title: title of the plot
    :return:
    """
    idx = image_y.argmax(axis=0)
    x, y = get_curve(image_y, y_pred, step_size)
    auc = evaluation_auc(image_y, y_pred, step_size)

#     plt.title(f'{title}: auc={auc}, y_true={idx}, voxel_count:{max_voxels:,} / {voxels:,})')
    plt.title(f'{title.capitalize()} (AUC = {auc / max_voxels:.2})')
    plt.plot(x, y, linewidth=2)

    ax = plt.gca()
    ax.set_ylabel(f'activation')
    ax.set_ylim(0, 1)
    ax.set_xlabel(f'voxels')


def predict_sequence_as_numpy(model, eval_seq, batch_size, log=False):
    """
    predict_sequence_as_numpy_v2
    eval_seq is sequence to evaluate
    batch_size is model.predict() batch size
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


def predict_sequence_as_numpy_v2(predict_fn, eval_seq, log=False):
    """
    predict_sequence_as_numpy_v2
    eval_seq is sequence to evaluate
    """
    y_pred = None
    count = len(eval_seq)

    for i, batch_x in enumerate(eval_seq):
        if log:
            print(f'evaluating batch {i}/{count} of length {len(batch_x)}...')
        batch_y_pred = predict_fn(batch_x)
        if y_pred is None:
            y_pred = batch_y_pred
        else:
            y_pred = np.concatenate([y_pred, batch_y_pred], axis=0)

    return y_pred
