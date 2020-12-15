from src.heatmaps.evaluation import plot_evaluation


class HeatmapEvaluationHistory:
    def __init__(self, method, auc, arr_auc, arr_heatmap, arr_x, arr_y, arr_y_pred, arr_y_pred_heatmap, arr_voxels, arr_max_voxels, arr_step_size):
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

    def plot_evaluation(self, idx):
        self.__ensure_idx(idx)
        return plot_evaluation(self.arr_y[idx], self.arr_y_pred[idx], self.arr_step_size[idx], self.arr_voxels[idx],
                               self.arr_max_voxels[idx], self.method)

    def __ensure_idx(self, idx):
        if idx >= len(self.arr_x):
            raise Exception(f'image with index {idx} does not exist in the history!')