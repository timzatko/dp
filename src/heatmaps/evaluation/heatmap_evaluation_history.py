class HeatmapEvaluationHistory:
    def __init__(self, auc, arr_auc, arr_heatmap, arr_x, arr_y, arr_y_pred, arr_y_pred_heatmap):
        self.auc = auc
        self.arr_auc = arr_auc
        self.arr_heatmap = arr_heatmap
        self.arr_x = arr_x
        self.arr_y = arr_y
        self.arr_y_pred = arr_y_pred
        self.arr_y_pred_heatmap = arr_y_pred_heatmap
