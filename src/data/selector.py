import numpy as np


def select_from_dataset(model, seq, max_category=5, **kwargs):
    images_x = []
    images_y = []

    tn = 0
    tp = 0
    fp = 0
    fn = 0
    
    for batch_x, batch_y in seq:
        batch_y_pred = np.argmax(model.predict(batch_x), axis=1)

        for y_pred, y_true_logits, x in zip(batch_y_pred, batch_y, batch_x):
            y_true = np.argmax(np.array(y_true_logits), axis=0)
            
            if y_pred == y_true == 1 and tp < kwargs.get('tp_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                tp += 1

            if y_pred == y_true == 0 and tn < kwargs.get('tn_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                tn += 1

            if y_pred != y_true and y_true == 0 and fp < kwargs.get('fp_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                fp += 1

            if y_pred != y_true and y_true == 1 and fn < kwargs.get('fn_max', max_category):
                images_x.append(x)
                images_y.append(y_true_logits)
                fn += 1

        if tn == kwargs.get('tn_max', max_category) and tp == kwargs.get('tp_max', max_category) and fp == kwargs.get('fp_max', max_category) and fn == kwargs.get('fn_max', max_category):
            break

    images_x = np.array(images_x).reshape((-1, *seq.input_shape))
    images_y = np.array(images_y)
    
    return images_x, images_y
    