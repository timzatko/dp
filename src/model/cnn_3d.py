import tensorflow as tf


# https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
def cnn_3d(
        input_shape=None,
        class_names=None,
        output_bias=None,
        batch_norm=False,
        is_complex=False,
        dropout=None,
        l2_beta=None):
    """
    input_shape is (z, x, y, 1)
    log_dir of the tensorboard logs
    class_names
    output_bias
    batch_norm
    is_complex
    """
    if input_shape is None:
        raise Exception("input_shape should not be none!")

    # In the original paper, they experiment with dropout and L2 regularizers
    # they do not specify, where they put dropout layers, and on which layers they
    # apply what types of regularizations
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=input_shape, name="input_layer"))

    # L1, L2
    # In the original paper they use input_shape=(116, 113, 83, 1), however it does not match
    # the proportions of our input shape
    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)
    model.add(tf.keras.layers.Conv3D(32, kernel_size=3, padding='same', kernel_regularizer=l2, input_shape=input_shape))
    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    if is_complex:
        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)
        model.add(tf.keras.layers.Conv3D(32, kernel_size=3, padding='same', kernel_regularizer=l2))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        
    # L3
    model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same'))

    # L4, L5
    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)
    model.add(tf.keras.layers.Conv3D(64, kernel_size=3, padding='same', kernel_regularizer=l2))
    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    if is_complex:
        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)
        model.add(tf.keras.layers.Conv3D(64, kernel_size=3, padding='same', kernel_regularizer=l2))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))

    # L6
    model.add(tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3)))

    # L7, L8
    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)
    model.add(tf.keras.layers.Conv3D(128, kernel_size=3, padding='same', kernel_regularizer=l2))
    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

    if is_complex:
        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)
        model.add(tf.keras.layers.Conv3D(128, kernel_size=3, padding='same', kernel_regularizer=l2))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))

    # L9
    model.add(tf.keras.layers.MaxPool3D(pool_size=(4, 4, 4)))

    # Flatten
    model.add(tf.keras.layers.Flatten())
    
    # Dropout
    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    # L10
    if is_complex:
        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2))

    # Dropout
    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    # L11
    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2))

    # https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    # Dropout
    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Output
    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model
