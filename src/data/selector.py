import numpy as np
import torch


def tf_predict(model):
    def fn(x):
        return model.predict(x)
    return fn


def torch_predict(model, batch_size=8):
    def fn(x):
        with torch.no_grad():
            y = model.eval()(torch.from_numpy(np.transpose(x, axes=(0, 4, 1, 2, 3))).float().to('cuda'))
            y_pred = y.to('cpu').detach().numpy()
        return y_pred
    return fn


def select_from_dataset(predict_fn, sequence, max_category=5, **kwargs):
    images_x = []
    images_y = []
    images_y_pred = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for batch_x, batch_y in sequence:
        batch_y_pred = predict_fn(batch_x)

        for y_pred, y_true_logits, x in zip(np.argmax(batch_y_pred, axis=1), batch_y, batch_x):
            y_true = np.argmax(np.array(y_true_logits), axis=0)
            
            if y_pred == y_true == 1 and tp < kwargs.get('tp_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                images_y_pred.append(y_pred)
                tp += 1

            if y_pred == y_true == 0 and tn < kwargs.get('tn_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                images_y_pred.append(y_pred)
                tn += 1

            if y_pred != y_true and y_true == 0 and fp < kwargs.get('fp_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                images_y_pred.append(y_pred)
                fp += 1

            if y_pred != y_true and y_true == 1 and fn < kwargs.get('fn_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                images_y_pred.append(y_pred)
                fn += 1

        if tn == kwargs.get('tn_max', max_category) and tp == kwargs.get('tp_max', max_category) and fp == kwargs.get('fp_max', max_category) and fn == kwargs.get('fn_max', max_category):
            break

    images_x = np.array(images_x).reshape((-1, *sequence.input_shape))
    images_y = np.array(images_y)
    
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    
    return images_x, images_y, images_y_pred
    