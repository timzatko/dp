import math

import matplotlib.pyplot as plt


def plot_training_history(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    metrics = list(filter(lambda m: 'val_' not in m, history.history.keys()))
    n_cols = 3
    n_rows = math.ceil(len(metrics) / n_cols)
    plt.figure(figsize=(n_rows * 6, n_rows * 6 * 1.25))

    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()

        plt.subplot(n_cols, n_rows, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.xlim([0, len(history.epoch)])

        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
