import io
import os
import sklearn

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from src.model.evaluation.plot_confusion_matrix import plot_confusion_matrix


def to_rgb_image(img, pred_label=None, true_label=None, z_index=None, add_batch_dim=True):
    figure = plt.figure(figsize=(4, 4))
    plt.imshow(img.reshape(img.shape[:-1])[z_index], cmap='gray')
    if true_label is not None and pred_label is not None:
        plt.title(f'true = {true_label}, pred = {pred_label}')
    return plot_to_image(figure, add_batch_dim)


def plot_to_image(figure, add_batch_dim=True):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    if add_batch_dim:
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
    return image


class MRITensorBoardCallback(Callback):
    def __init__(self, seq, model, z_index=56, max_outputs=18, freq=3, log_dir=None, debug=True):
        """
        seq is the sequence from which is the data taken
        model to fit
        z_index is the index in the 3D image which is visualised
        log_dir is the where is output logged
        max_outputs number of images to output
        freq determines how frequently (each freq epoch) to outpu to tensorboard
        """
        super(MRITensorBoardCallback, self).__init__()
        self.model = model
        self.seq = seq
        self.log_dir = log_dir
        self.z_index = z_index
        self.max_outputs = max_outputs
        self.freq = freq
        self.debug = debug

    def __get_z_index(self, img):
        return max(min(img.shape[1], self.z_index), 0)

    def __debug(self, msg):
        if self.debug:
            print(msg)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq != 0:
            self.__debug('skipping evaluation of predictions to tensorboard')
            return

        self.__debug(
            f'evaluation of predictions to tensorboard for epoch #{epoch} (no of batches is {len(self.seq)})...')

        images = []
        class_names = self.seq.class_names
        y_pred = np.array([]).reshape(-1, len(class_names))
        y_true = np.array([]).reshape(-1, len(class_names))

        # Get predictions in batches for seq
        for index, batch in enumerate(self.seq):
            x, y, _ = batch
            self.__debug(f'batch #{index}')
            # Get predictions
            pred = self.model.predict(x)

            # Merge with other predictions
            y_true = np.concatenate([y_true, y])
            y_pred = np.concatenate([y_pred, pred])

            # Encode labels
            true_labels = self.seq.encoder.inverse_transform(y)
            pred_labels = self.seq.encoder.inverse_transform(pred)

            # Do not create more images than we output
            if len(images) >= self.max_outputs:
                continue;

            rgb_images = [
                to_rgb_image(image, pred_label=pred, true_label=true, z_index=self.z_index, add_batch_dim=False) for
                image, pred, true in zip(x, pred_labels.reshape(-1), true_labels.reshape(-1))]
            for rgb_image in rgb_images:
                images.append(rgb_image)

        # Create a confusion matrix
        cm = sklearn.metrics.confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(os.path.join(self.log_dir, 'validation/confussion-matrix'))
        file_writer_images = tf.summary.create_file_writer(os.path.join(self.log_dir, 'validation/images'))

        with file_writer_images.as_default():
            # Don't forget to reshape.
            images = images[0:self.max_outputs]
            tf.summary.image("Validation Images", images, max_outputs=self.max_outputs, step=epoch)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)