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
        return im[:, y, :]
    if x is not None:
        return im[:, :, x]
    raise "x, y or z must be defined!"

    
def dim_selector_factory(x=None, y=None, z=None):
    def fn(im):
        return dim_selector(im, x=x, y=y, z=z)
    return fn


def get_edges(img):
    return filters.roberts(img.reshape(img.shape[:-1])).reshape(img.shape)

import skimage

def rotate(im, angle):
    if angle is None:
        return im
    for _ in range(angle):
        im = np.rot90(im)
    return im


def plot_segmentation(seg_mask, seg_label, image_x, image_y, heatmap, d_selector, show_image=True, show_heatmap=True, edges=True, alpha=0.5, hmap_imshow=None, rotate_angle=None, ax=None, legend=True):
    cls = ['red', 'red', 'black', 'magenta', 'brown']
    
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        
    handles = []

    if show_heatmap:
        if hmap_imshow is None:
            ax.imshow(rotate(d_selector(heatmap), rotate_angle), alpha=1 - alpha, cmap='jet')
        else:
            hmap_imshow(ax, rotate(d_selector(heatmap), rotate_angle), 1 - alpha)
    
    if show_image:
        ax.imshow(rotate(d_selector(image_x), rotate_angle), alpha=alpha, cmap='viridis')
    
    for idx, label in seg_label.items():
        if label == 'background / non-brain':
            continue
        color = cls[idx]
        handles.append(mpatches.Patch(color=color, label=lnames[label]))
        new_seg_mask = d_selector(seg_mask) == idx
        new_seg_mask = new_seg_mask.astype(np.int64)
        if edges:
            new_seg_mask = get_edges(new_seg_mask)
        cmap = colors.ListedColormap([color, color])
        norm = colors.BoundaryNorm([0, 1], cmap.N)
        masked = np.ma.masked_where(new_seg_mask == 0, new_seg_mask)
        ax.imshow(rotate(masked, rotate_angle), alpha=1, cmap=cmap, norm=norm)

    if legend:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.225, 1., 0., 0.))
    
    return ax, handles

lnames = {
    'background / non-brain': 'nie mozog',
    'hippocampus': 'hipokampus',
    'ventricles': 'komory',
    'gray_matter': 'šedá hmota',
    'white matter': 'biela hmota',
}

import tensorflow.keras.backend as K

# from https://github.com/keras-team/keras/blob/c10d24959b0ad615a21e671b180a1b2466d77a2b/examples/conv_filter_visualization.py
def norm(arr):
    x = np.array(arr).copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    return x


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
    
        self.heatmaps = self.history_ins.arr_heatmap
        
        self.fkey = fkey
        
        self.images = list(zip(self.seg_masks, self.seg_labels, self.images_x, self.images_y, self.heatmaps, self.history_ins.arr_y_pred))
        
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
            # areas_heat_sum_density - size normalized
            areas_heat_sum_density = {}

            hmap = heatmap.flatten()
            # hmap = np.array([x if x > 0 else 0 for x in hmap]) # remove the negative heat
            hmap = norm(hmap)
            
            def get_ratio(a_labels, b_labels):
                heat_sum_a = sum([np.sum(heat_sum_values) for seg_label, heat_sum_values in areas_heat.items() if seg_label in a_labels])
                area_a = sum([area_size for seg_label, area_size in areas.items() if seg_label in a_labels])
                
                heat_sum_b = sum([np.sum(heat_sum_values) for seg_label, heat_sum_values in areas_heat.items() if seg_label in b_labels])
                area_b = sum([area_size for seg_label, area_size in areas.items() if seg_label in b_labels])
                
                return (heat_sum_a / area_a) / (heat_sum_b / area_b)

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
                areas_heat_sum_density[seg_label] = heat_sum / areas[seg_label]
                
            row = {}
            y_true = np.argmax(image_y, axis=0)
            add_to("y_true", row, y_true)
            add_to("y_pred", row, abs(y_true - image_y_pred[y_true]))
            for seg_label, heat_sum in areas_heat_sum.items():
                add_to(f"arr_heat_sum__{seg_label}", row, heat_sum)
            for seg_label, heat_sum in areas_heat_sum_density.items():
                add_to(f"arr_heat_sum_density__{seg_label}", row, heat_sum)
                
            add_to(f"arr_heat_sum_non_brain_vs_brain", row, get_ratio([0], [1, 2, 3, 4]))
            add_to(f"arr_heat_sum_0_2_vs_1_3_4", row, get_ratio([0, 2], [1, 3, 4]))
            add_to(f"arr_heat_sum_0_vs_1_3_4", row, get_ratio([0], [1, 3, 4]))
            add_to(f"arr_heat_sum_0_vs_3_4", row, get_ratio([0], [3, 4]))
            add_to(f"arr_heat_sum_0_1_2_vs_3_4", row, get_ratio([0], [1, 2, 3, 4]))

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
        
        def describe(df, cls_idx, suffix):
            for row in df.describe().itertuples():
                idx = None
                for key, value in row._asdict().items():
                    if key == 'Index':
                        idx = value
                        continue
                    if idx == 'count':
                        continue
                    new_key = f'{key}__{idx}__{suffix}'
                    new_row[new_key] = value

            hist_ins_desc = self.history_ins._description(cls_index=cls_idx)
            for key, value in hist_ins_desc.items():
                new_row[f'insertion__{key}__{suffix}'] = value

            hist_del_desc = self.history_del._description(cls_index=cls_idx)
            for key, value in hist_del_desc.items():
                new_row[f'deletion__{key}__{suffix}'] = value
                
        describe(self.df, None, 'AD+CN')
        describe(self.df[self.df['y_true'] == 0], 0, 'AD')
        describe(self.df[self.df['y_true'] == 1], 1, 'CN')
            
        return new_row
    
    
class SegmentationMasksSaver(CSVSaver):
    def __init__(self, root_dir, fname="evaluation.csv"):
        super().__init__(os.path.join(root_dir, fname), "notebook_key")