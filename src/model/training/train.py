import os
import datetime

import tensorflow as tf

from src.data.augmentations import augment, get_class_weight
from src.data.sequence_to_numpy import sequence_to_numpy
from src.model.training.mri_tensorboard_callback import MRITensorBoardCallback


def train(model,
          train_seq,
          val_seq,
          checkpoint_directory,
          log_directory,
          data_directory,
          epochs=50,
          patience=10,
          model_key=None,
          tensorboard_update_freq='epoch',
          mri_tensorboard_callback=False,
          model_checkpoint_callback=True,
          model_checkpoint_callback_monitor='val_auc',
          early_stopping_monitor='val_auc',
          augmentations=True,
          batch_size=8):
    """
    Start training the model.
    """
    input_shape = train_seq.input_shape

    if model_key is None:
        model_key = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    print(f'model key: {model_key}')

    checkpoint_dir = os.path.join(checkpoint_directory, model_key)
    log_dir = os.path.join(log_directory, model_key)

    print(f'checkpoint dir: {checkpoint_dir}')
    print(f'log dir: {log_dir}')

    callbacks = [
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor,
            patience=patience,  # Number of epochs with no improvement after which training will be stopped.
            restore_best_weights=True,
        ),
    ]

    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq=tensorboard_update_freq,  # batch frequency number / 'epoch'
        histogram_freq=0,
        profile_batch=0
    )),

    if model_checkpoint_callback is not False:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt'),
            monitor=model_checkpoint_callback_monitor,
            save_weights_only=True,
            verbose=2,
            save_best_only=model_checkpoint_callback == 'save_best_only'
        )),

    if mri_tensorboard_callback:
        z_index = input_shape[0] // 2
        callback = MRITensorBoardCallback(val_seq, model, z_index=z_index, freq=1, log_dir=log_dir, debug=False)
        callbacks.append(callback)

    train_x, train_y = sequence_to_numpy(train_seq, data_directory, 'train')
    val_x, val_y = sequence_to_numpy(val_seq, data_directory, 'val')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    if augmentations:
        train_dataset = train_dataset.map(lambda x, y: (
            tf.py_function(func=augment, inp=[x], Tout=tf.float32),
            y, tf.py_function(func=get_class_weight, inp=[y], Tout=tf.float32)))
    else:
        train_dataset = train_dataset.map(lambda x, y: (
            x,
            y,
            tf.py_function(func=get_class_weight, inp=[y],
                           Tout=tf.float32)))

    # create batches from train dataset
    batched_train = train_dataset.batch(batch_size)

    # create val dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_dataset = val_dataset.map(lambda x, y: (x, y, tf.py_function(func=get_class_weight, inp=[y], Tout=tf.float32)))
    # create batches from val dataset
    batched_val = val_dataset.batch(batch_size)

    # train the model
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    print('training...')
    history = model.fit(
        batched_train,
        validation_data=batched_val,
        epochs=epochs,
        # class_weight=class_weight,
        callbacks=callbacks)

    return model, checkpoint_dir, history
