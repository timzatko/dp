import tensorflow as tf


# https://link.springer.com/chapter/10.1007/978-3-319-75417-8_27
def deep_3d(output_bias, class_names, input_shape=(96, 96, 64), dropout=0.7, batch_norm=False):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape, name='InputLayer'))
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 5), strides=2))
    if batch_norm is not None:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

    model.add(BasicBlock(16, batch_norm=batch_norm))
    model.add(BasicBlock(16, batch_norm=batch_norm))
    model.add(BasicBlock(16, batch_norm=batch_norm, strides=(2, 2, 2)))

    model.add(BasicBlock(32, batch_norm=batch_norm))
    model.add(BasicBlock(32, batch_norm=batch_norm))
    model.add(BasicBlock(32, batch_norm=batch_norm))
    model.add(BasicBlock(32, batch_norm=batch_norm, strides=(2, 2, 2)))

    model.add(BasicBlock(64, batch_norm=batch_norm))
    model.add(BasicBlock(64, batch_norm=batch_norm))
    model.add(BasicBlock(64, batch_norm=batch_norm))
    model.add(BasicBlock(64, batch_norm=batch_norm))
    model.add(BasicBlock(64, batch_norm=batch_norm))
    model.add(BasicBlock(64, batch_norm=batch_norm, strides=(2, 2, 2)))

    model.add(BasicBlock(128, batch_norm=batch_norm))
    model.add(BasicBlock(128, batch_norm=batch_norm))
    model.add(BasicBlock(128, batch_norm=batch_norm))
    model.add(BasicBlock(128, batch_norm=batch_norm, strides=(2, 2, 2)))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.GlobalAveragePooling3D())
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, batch_norm=False, strides=None):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = tf.keras.layers.Conv3D(
            filters=filter_num,
            kernel_size=(3, 3),
            padding="same"
        )

        if batch_norm is not None:
            self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv3D(
            filters=filter_num,
            kernel_size=(3, 3),
            padding="same",
            strides=(1, 1, 1) if strides is None else strides
        )

        if batch_norm is not None:
            self.bn1 = tf.keras.layers.BatchNormalization()

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
