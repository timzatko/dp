import time

import numpy as np

from src.heatmaps.evaluation.evaluation_sequence import EvaluationSequence
from src.heatmaps.evaluation.utils import evaluation_auc, predict_sequence_as_numpy
from src.heatmaps.evaluation.heatmap_evaluation_history import HeatmapEvaluationHistory
from src.heatmaps.heatmaps import get_heatmap


class HeatmapEvaluation:
    def __init__(self, risei, model, seq, batch_size,
                 masks_count=120,
                 evaluation_step_size=1000,
                 evaluation_max_steps=-1,
                 risei_batch_size=480,
                 evaluation_batch_size=32):
        """
        Generate and evaluate heatmaps for MRI images.
        :param model: model used for heatmap generation and evaluation
        :param seq: sequence with images (val_seq, train_seq, test_seq...)
        :param batch_size: batch_size for model (depends on GPU)
        :param masks_count: how many masks to generate for heatmap evaluation
        :param evaluation_step_size: step size when evaluating heatmap, see: EvaluationSequence
        :param evaluation_max_steps: maximum number of steps when evaluating heatmap, see: EvaluationSequence
        :param risei_batch_size: batch size when generating heatmap (depends on RAM)
        :param evaluation_batch_size:
        """
        self.model = model
        self.risei = risei
        self.seq = seq
        self.batch_size = batch_size
        self.masks_count = masks_count
        self.evaluation_step_size = evaluation_step_size
        self.evaluation_max_steps = evaluation_max_steps
        self.risei_batch_size = risei_batch_size
        self.evaluation_batch_size = evaluation_batch_size
        self.cache = None

    def evaluate(self, method='deletion', log=False, verbose=0, seed=None, max_evaluations=None):
        """
        Evaluate sequence with provided method.
        :param max_evaluations: maximum number of evaluated images
        :param method:
        :param log:
        :param verbose:
        :param seed: seed for the heatmap generation, None means no seed will be applied. If seed is applied, for each
        two run's kth images will have same masks generated.
        :return:
        """
        evaluations = 0
        length = len(self.seq.images_dirs)
        arr_auc = []
        arr_heatmap = []
        arr_x = []
        arr_y = []
        arr_y_pred = []
        arr_y_pred_heatmap = []

        print(f'sequence len: {length}, method: {method}')
        for batch_x, batch_y, *_ in self.seq:
            batch_y_pred = self.model.predict(batch_x)
            
            if max_evaluations is not None and max_evaluations == evaluations + 1:
                if log:
                    print(f'max evaluations ({max_evaluations}) reached!')
                break

            for i, image in enumerate(zip(batch_x, batch_y)):
                if max_evaluations is not None and max_evaluations == evaluations + 1:
                    break

                image_x, image_y = image
                y_pred = batch_y_pred[i]
                start = time.time()

                if log:
                    print(f'evaluation {evaluations + 1}/{length}')
                    print(f'get heatmap (masks: {self.masks_count})...')

                start_a = time.time()
                # get heatmap for the image
                heatmap, _, _ = get_heatmap(
                    image_x,
                    image_y,
                    self.model,
                    self.risei,
                    batch_size=self.batch_size,
                    masks_count=self.masks_count,
                    risei_batch_size=self.risei_batch_size,
                    debug=False,
                    seed=None if seed is None else seed + evaluations,
                    log=log and verbose > 1
                )
                end_a = time.time()
                print(f'...finished in {end_a - start_a}s')

                start_a = time.time()
                # create a sequence of images, by removing/insering voxels
                # from the image, and evaluate a them on the model
                eval_seq = EvaluationSequence(
                    method,
                    image_x,
                    heatmap,
                    step_size=self.evaluation_step_size,
                    max_steps=self.evaluation_max_steps,
                    batch_size=self.evaluation_batch_size,
                    debug=False,
                    log=log and verbose > 1
                )

                if log:
                    vx = np.sum(image_x.shape) if eval_seq.max_steps < 0 else eval_seq.step_size * eval_seq.max_steps
                    print(
                        f'evaluate heatmaps (voxels: {vx}, '
                        f'step_size: {self.evaluation_step_size}, max_steps: {self.evaluation_max_steps})...')

                y_pred_heatmap = predict_sequence_as_numpy(self.model, eval_seq, self.batch_size, log=verbose > 1)
                end_a = time.time()
                if log:
                    print(f'...finished in {end_a - start_a}s')
                auc = evaluation_auc(image_y, y_pred_heatmap, eval_seq.step_size)

                arr_heatmap.append(heatmap)
                arr_x.append(image_x)
                arr_y.append(image_y)
                arr_y_pred_heatmap.append(y_pred_heatmap)
                arr_y_pred.append(y_pred)
                arr_auc.append(auc)
                evaluations += 1

                end = time.time()

                if log:
                    print(f'auc: {auc} ({end - start}s)')
                    print()

        auc = sum(arr_auc) / evaluations

        return HeatmapEvaluationHistory(auc, arr_auc, arr_heatmap, arr_x, arr_y, arr_y_pred, arr_y_pred_heatmap)
