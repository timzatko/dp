import tensorflow as tf

from tensorflow.keras.applications import ResNet50V2


def res_net_50_v2(input_shape, class_names, batch_norm=None, l2_beta=None, dropout=None, output_bias=None):
    input_layer = tf.keras.layers.Input(shape=input_shape, name='InputLayer')
    reshape_layer = tf.keras.layers.Reshape(input_shape[:-1])(input_layer)

    core = ResNet50V2(
        input_tensor=reshape_layer,
        weights=None,
        input_shape=input_shape[:-1],
        pooling='max',
        classes=1024,
    )
    core.trainable = True

    # add regularization to layers of res net
    if l2_beta is not None:
        for layer in core.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, tf.keras.regularizers.l2(l2_beta))

    model = tf.keras.models.Sequential()
    model.add(core)

    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)

    model.add(tf.keras.layers.Dense(512, kernel_regularizer=l2))

    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)

    model.add(tf.keras.layers.Dense(256, kernel_regularizer=l2))

    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model
