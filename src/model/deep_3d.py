import tensorflow as tf


# https://link.springer.com/chapter/10.1007/978-3-319-75417-8_27
def deep_3d(output_bias, class_names, input_shape=None, l2_beta=None, dropout=0.7, batch_norm=False):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape, name='InputLayer'))
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 5), strides=2, padding='same'))
    if batch_norm is not None:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    model.add(BasicBlock(16, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(16, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(16, l2_beta=l2_beta, batch_norm=batch_norm, strides=(2, 2, 2)))

    model.add(BasicBlock(32, l2_beta=l2_beta, batch_norm=batch_norm, up_sample=True))
    model.add(BasicBlock(32, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(32, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(32, l2_beta=l2_beta, batch_norm=batch_norm, strides=(2, 2, 2)))

    model.add(BasicBlock(64, l2_beta=l2_beta, batch_norm=batch_norm, up_sample=True))
    model.add(BasicBlock(64, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(64, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(64, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(64, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(64, l2_beta=l2_beta, batch_norm=batch_norm, strides=(2, 2, 2)))

    model.add(BasicBlock(128, l2_beta=l2_beta, batch_norm=batch_norm, up_sample=True))
    model.add(BasicBlock(128, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(128, l2_beta=l2_beta, batch_norm=batch_norm))
    model.add(BasicBlock(128, l2_beta=l2_beta, batch_norm=batch_norm))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.GlobalAveragePooling3D())

    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)

    model.add(tf.keras.layers.Dense(128, kernel_regularizer=l2, activation=tf.keras.activations.relu))

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, batch_norm=False, l2_beta=None, strides=None, up_sample=False):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm

        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)

        self.conv1 = tf.keras.layers.Conv3D(
            filters=filter_num,
            kernel_size=(3, 3, 3),
            padding='same',
            kernel_regularizer=l2,
        )

        if batch_norm is not None:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv3D(
            filters=filter_num,
            kernel_size=(3, 3, 3),
            padding='same',
            strides=(1, 1, 1) if strides is None else strides,
            kernel_regularizer=l2,
        )

        if batch_norm is not None:
            self.bn2 = tf.keras.layers.BatchNormalization()

        if strides is not None:
            self.residual = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(1, 1, 1), strides=strides)
        elif up_sample:
            self.residual = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(1, 1, 1))
        else:
            self.residual = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.residual(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
