from src.heatmaps import plot_heatmap_x, plot_heatmap_y, plot_heatmap_z
from src.heatmaps.evaluation.utils import plot_evaluation

import numpy as np
import seaborn as sns

import os
import pickle


class HeatmapEvaluationHistory:
    def __init__(self, method, auc, arr_auc, arr_heatmap, arr_x, arr_y, arr_y_pred, arr_y_pred_heatmap,
                 arr_voxels, arr_max_voxels, arr_step_size):
        """

        :param method:
        :param auc:
        :param arr_auc:
        :param arr_heatmap:
        :param arr_x:
        :param arr_y:
        :param arr_y_pred:
        :param arr_y_pred_heatmap: predictions (activations) for images generated by evaluation sequence
        :param arr_voxels:
        :param arr_max_voxels:
        :param arr_step_size:
        """
        self.method = method
        self.auc = auc
        self.arr_auc = arr_auc
        self.arr_heatmap = arr_heatmap
        self.arr_x = arr_x
        self.arr_y = arr_y
        self.arr_y_pred = arr_y_pred
        self.arr_y_pred_heatmap = arr_y_pred_heatmap
        self.arr_voxels = arr_voxels
        self.arr_max_voxels = arr_max_voxels
        self.arr_step_size = arr_step_size

    @staticmethod
    def load(path, filename):
        p = os.path.join(path, f'{filename}.cls')
        with open(p, 'rb') as file:
            return pickle.load(file)

    def save(self, path, filename):
        p = os.path.join(path, f'{filename}.cls')
        if not os.path.exists(path):
            os.mkdir(path)
        with open(p, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saved to: {p}')

    def description(self):
        print(f'auc')
        print(f'\tmean: {np.mean(self.auc):,}')
        print(f'\tmedian: {np.median(self.auc):,}')
        print(f'\tmax: {np.max(self.auc):,}')
        print(f'\tmin: {np.min(self.auc):,}')
        print(f'\tstd: {np.std(self.auc):,}')

    def plot_auc(self):
        f_grid = sns.displot(self.arr_auc)
        f_grid.set_axis_labels('AUC (area under curve)')
        f_grid.axes[0][0].set_title('Distribution of AUC metric for generated heatmaps')
        return f_grid

    def list_auc(self, round_auc=False):
        arr = np.sort(np.array(self.arr_auc))
        for i, auc in enumerate(arr):
            print(f'idx: {i}, auc: {round(auc) if round_auc else auc:,}')

    def plot_evaluation(self, idx):
        self.__ensure_idx(idx)
        return plot_evaluation(self.arr_y[idx], self.arr_y_pred_heatmap[idx], self.arr_step_size[idx],
                               self.arr_voxels[idx],
                               self.arr_max_voxels[idx], self.method)

    def plot_heatmap_x(self, idx, i=None):
        self.__ensure_idx(idx)
        plot_heatmap_x(self.arr_x[idx], self.arr_y[idx], self.arr_y_pred[idx], self.arr_heatmap[idx], i)

    def plot_heatmap_y(self, idx, i=None):
        self.__ensure_idx(idx)
        plot_heatmap_y(self.arr_x[idx], self.arr_y[idx], self.arr_y_pred[idx], self.arr_heatmap[idx], i)

    def plot_heatmap_z(self, idx, i=None):
        self.__ensure_idx(idx)
        plot_heatmap_z(self.arr_x[idx], self.arr_y[idx], self.arr_y_pred[idx], self.arr_heatmap[idx], i)

    def __ensure_idx(self, idx):
        if idx >= len(self.arr_x):
            raise Exception(f'image with index {idx} does not exist in the history!')
