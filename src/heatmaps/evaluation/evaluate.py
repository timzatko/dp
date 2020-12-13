import time

import numpy as np

from src.heatmaps.evaluation.evaluation import EvaluationSequence, predict_sequence_as_numpy, evaluation_auc
from src.heatmaps.heatmaps import get_heatmap


def evaluate_sequence(model,
                      risei,
                      seq,
                      batch_size,
                      masks_count=120,
                      evaluation_step_size=1000,
                      evaluation_max_steps=1000,
                      risei_batch_size=480,
                      evaluation_batch_size=32,
                      method='deletion',
                      log=False,
                      verbose=0):
    evaluations = 0
    length = len(seq.images_dirs)
    arr_auc = []
    arr_heatmap = []
    arr_x = []
    arr_y = []
    arr_y_pred = []
    arr_y_pred_heatmap = []

    print(f'sequence len: {length}, method: {method}')
    for batch_x, batch_y, *_ in seq:
        batch_y_pred = model.predict(batch_x)

        for i, image in enumerate(zip(batch_x, batch_y)):
            image_x, image_y = image
            y_pred = batch_y_pred[i]
            start = time.time()

            if log:
                print(f'evaluation {evaluations + 1}/{length}')
                print(f'get heatmap (masks: {masks_count})...')

            start_a = time.time()
            heatmap, _, _ = get_heatmap(
                image_x,
                image_y,
                model,
                risei,
                batch_size=batch_size,
                masks_count=masks_count,
                risei_batch_size=risei_batch_size,
                debug=False,
                log=log and verbose > 1
            )
            end_a = time.time()
            print(f'...finished in {end_a - start_a}s')

            start_a = time.time()
            eval_seq = EvaluationSequence(
                method,
                image_x,
                heatmap,
                step_size=evaluation_step_size,
                max_steps=evaluation_max_steps,
                batch_size=evaluation_batch_size,
                debug=False,
                log=log and verbose > 1
            )

            if log:
                print(
                    f'evaluate heatmaps (voxels: {eval_seq.max_steps * eval_seq.step_size}, step_size: {evaluation_step_size}, max_steps: {evaluation_max_steps})...')

            y_pred_heatmap = predict_sequence_as_numpy(model, eval_seq, batch_size, log=verbose > 1)
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

    return auc, arr_auc, arr_heatmap, arr_x, arr_y, arr_y_pred, arr_y_pred_heatmap


def evaluate_sequence_heatmap(idx, fn, arr_heatmap, arr_x, arr_y, arr_y_pred, arr_y_pred_heatmap):
    if arr_y_pred_heatmap is not None:
        print(f'y_pred_heatmap: {np.average(arr_y_pred[idx], axis=0)}')
    return fn(arr_x[idx], arr_y[idx], arr_y_pred[idx], arr_heatmap[idx], 56)
