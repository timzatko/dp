from src.data import select_from_dataset
from src.data import tf_predict
from src.heatmaps.evaluation import HeatmapEvaluationHistory


from .saver import CSVSaver

import os

import pandas as pd
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from skimage import filters


def get_segmentation_masks_factory(model, sequence, max_category=20, fp_max=0, fn_max=0, batch_size=8):
    def fn():
        images_x, images_y, images_y_pred, (seg_masks, seg_labels) = select_from_dataset(tf_predict(model), sequence, max_category=max_category, fp_max=fp_max, fn_max=fn_max, seq_other=True)
        print(images_x.shape)
        return images_x, images_y, images_y_pred, seg_masks, seg_labels
    return fn


def dim_selector(im, x=None, y=None, z=None):
    if z is not None:
        return im[z, :, :]
    if y is not None:
        return im[:, y: :]
    if x is not None:
        return im[:, :, x]
    raise "x, y or z must be defined!"

    
def dim_selector_factory(x=None, y=None, z=None):
    def fn(im):
        return dim_selector(im, x=x, y=y, z=z)
    return fn


def get_edges(img):
    return filters.roberts(img.reshape(img.shape[:-1])).reshape(img.shape)


def plot_segmentation(seg_mask, seg_label, image_x, image_y, heatmap, d_selector, show_image=True, show_heatmap=True, edges=True, alpha=0.5, hmap_imshow=None):
    cls = ['black', 'red', 'blue', 'magenta', 'brown']
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    handles = []

    if show_heatmap:
        if hmap_imshow is None:
            ax.imshow(d_selector(heatmap), alpha=1 - alpha, cmap='jet')
        else:
            hmap_imshow(ax, d_selector(heatmap), 1 - alpha)
    
    if show_image:
        ax.imshow(d_selector(image_x), alpha=alpha, cmap='viridis')
    
    for idx, label in seg_label.items():
        if label == 'background / non-brain':
            continue
        color = cls[idx]
        handles.append(mpatches.Patch(color=color, label=label))
        new_seg_mask = d_selector(seg_mask) == idx
        new_seg_mask = new_seg_mask.astype(np.int64)
        if edges:
            new_seg_mask = get_edges(new_seg_mask)
        cmap = colors.ListedColormap([color, color])
        norm = colors.BoundaryNorm([0, 1], cmap.N)
        masked = np.ma.masked_where(new_seg_mask == 0, new_seg_mask)
        ax.imshow(masked, alpha=1, cmap=cmap, norm=norm)

    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.225, 1., 0., 0.))
    
    return fig


def scale(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


class SegmentationMasksEvaluator():
    def __init__(self, segmentation_masks_factory, hisotry_fname_factory):
        images_x, images_y, images_y_pred, seg_masks, seg_labels = segmentation_masks_factory()
        
        self.images_x = images_x
        self.images_y = images_y
        self.images_y_pred = images_y_pred
        self.seg_masks = seg_masks
        self.seg_labels = seg_labels
       
        fpath, fkey, fname_ins, fname_del = hisotry_fname_factory()
        self.history_ins = HeatmapEvaluationHistory.load(fpath, fname_ins)
        self.history_del = HeatmapEvaluationHistory.load(fpath, fname_del)
        
        self.fkey = fkey
        
        self.images = list(zip(self.seg_masks, self.seg_labels, self.images_x, self.images_y, self.history_ins.arr_heatmap, self.history_ins.arr_y_pred))
        
        self.validate()
        self.evaluate()


    def evaluate(self):
        column_values = []
        column_names = []
        
        def add_to(column, obj, value):
            if column not in column_names:
                column_names.append(column)
            obj.update({ column: value })

        for seg_mask, seg_label, image_x, image_y, heatmap, image_y_pred in self.images:
            # areas -> key is seg_label, value is number of voxels
            areas = {}
            # areas_heat -> keys is seg_label, value is array of "heats"
            areas_heat = {}
            # areas_heat_sum
            areas_heat_sum = {}
            # areas_heat_sum_norm - size normalized
            areas_heat_sum_norm = {}
            # gain compared to background / non-brain
            areas_heat_gain = {}

            hmap = heatmap.flatten()
            hmap = np.array([x if x > 0 else 0 for x in hmap]) # remove the negatove heat
            hmap = scale(hmap)

            for voxel_seg_mask, voxel_heat in zip(seg_mask.flatten().astype(np.int64), hmap):
                if voxel_seg_mask not in areas:
                    areas[voxel_seg_mask] = 0
                areas[voxel_seg_mask] += 1

                if voxel_seg_mask not in areas_heat:
                    areas_heat[voxel_seg_mask] = []
                areas_heat[voxel_seg_mask].append(voxel_heat)

            for seg_label, heat_sum_values in areas_heat.items():
                areas_heat_sum[seg_label] = np.sum(heat_sum_values)

            for seg_label, heat_sum in areas_heat_sum.items():
                areas_heat_sum_norm[seg_label] = heat_sum / areas[seg_label]

            for seg_label, heat_sum_norm in areas_heat_sum_norm.items():
                if areas_heat_sum_norm[0] > 0:
                    areas_heat_gain[seg_label] = heat_sum_norm / areas_heat_sum_norm[0]
                else:
                    areas_heat_gain[seg_label] = np.nan

            row = {}
            y_true = np.argmax(image_y, axis=0)
            add_to("y_true", row, y_true)
            add_to("y_pred", row, abs(y_true - image_y_pred[y_true]))
            for seg_label, heat_sum in areas_heat_sum.items():
                add_to(f"arr_heat_sum__{seg_label}", row, heat_sum)
            for seg_label, heat_sum in areas_heat_sum_norm.items():
                add_to(f"arr_heat_sum_norm__{seg_label}", row, heat_sum)
            for seg_label, heat_gain in areas_heat_gain.items():
                add_to(f"arr_heat_sum_gain__{seg_label}", row, heat_gain)

            heat_sum_other = sum([np.sum(heat_sum_values) for seg_label, heat_sum_values in areas_heat.items() if seg_label != 0])
            area_other = sum([area_size for seg_label, area_size in areas.items() if seg_label != 0])
            heat_sum_norm_other = (heat_sum_other / area_other)
            heat_gain_other = np.nan if areas_heat_sum_norm[0] == 0 else heat_sum_norm_other / areas_heat_sum_norm[0]
            add_to(f"arr_heat_sum_gain_other", row, heat_gain_other)

            column_values.append(list(row.values()))

        self.df = pd.DataFrame(np.array(column_values), columns=column_names)
    
    def validate(self):
        s = 0
        for im_x1, im_x2 in zip(self.history_ins.arr_x, self.images_x):
            s += np.sum(im_x1 - im_x2)
        if s > 0:
            raise Exception("evaluation history (ins) does not match input images")
        
        s = 0
        for im_x1, im_x2 in zip(self.history_del.arr_x, self.images_x):
            s += np.sum(im_x1 - im_x2)
        if s > 0:
            raise Exception("evaluation history (del) does not match input images")
            
    def to_row(self):
        new_row = { 'notebook_key': self.fkey }
        for row in self.df.describe().itertuples():
            idx = None
            for key, value in row._asdict().items():
                if key == 'Index':
                    idx = value
                    continue
                if idx == 'count':
                    continue
                new_key = f'{key}__{idx}'
                new_row[new_key] = value
        
        hist_ins_desc = self.history_ins._description()
        for key, value in hist_ins_desc.items():
            new_row[f'insertion__{key}'] = value
            
        hist_del_desc = self.history_del._description()
        for key, value in hist_del_desc.items():
            new_row[f'deletion__{key}'] = value
            
        return new_row
    
    
class SegmentationMasksSaver(CSVSaver):
    def __init__(self, root_dir):
        super().__init__(os.path.join(root_dir, "evaluation.csv"), "notebook_key")