import numpy as np
import sklearn

from .plot_confusion_matrix import plot_confusion_matrix
from .custom_classification_report import custom_classification_report


def show_metrics(model, test_seq, class_names):
    y_true = np.array([]).reshape(-1, len(class_names))
    y_pred = np.array([]).reshape(-1, len(class_names))

    for batch in test_seq:
        x, y = batch

        pred = model.predict(x)

        y_true = np.concatenate([y_true, y])
        y_pred = np.concatenate([y_pred, pred])

    y_true_labels = test_seq.encoder.transform(test_seq.encoder.inverse_transform(y_true).reshape(-1, 1))
    y_pred_labels = test_seq.encoder.transform(test_seq.encoder.inverse_transform(y_pred).reshape(-1, 1))

    # Plot the confusion matrix
    cm = sklearn.metrics.confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plot_confusion_matrix(cm, class_names)

    # Plot the metrics
    custom_classification_report(class_names, y_true_labels, y_pred_labels)