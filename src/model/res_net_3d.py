import tensorflow as tf


# https://github.com/calmisential/TensorFlow2.0_ResNets
def res_net_3d(input_shape, class_names, l2_beta=None, dropout=None, output_bias=None, blocks=(2, 2, 2, 2),
               filters=(64, 128, 256, 512)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape, name='InputLayer'))
    model.add(res_net_18(
        classes=256,
        l2_beta=l2_beta,
        blocks=blocks,
        filters=filters
    ))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)

    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, l2_beta=None, stride=1):
        super(BasicBlock, self).__init__()
        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(3, 3, 3),
                                            strides=stride,
                                            kernel_regularizer=l2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(3, 3, 3),
                                            kernel_regularizer=l2,
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.down_sample = tf.keras.Sequential()
            self.down_sample.add(tf.keras.layers.Conv3D(filters=filter_num,
                                                        kernel_size=(1, 1, 1),
                                                        kernel_regularizer=l2,
                                                        strides=stride))
            self.down_sample.add(tf.keras.layers.BatchNormalization())
        else:
            self.down_sample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.down_sample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class MyResNet(tf.keras.Model):
    def __init__(self, layer_params, filter_params, classes, l2_beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = tf.keras.layers.Conv3D(filters=64,
                                            kernel_size=(7, 7, 7),
                                            strides=2,
                                            padding="same")

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3),
                                               strides=2,
                                               padding="same")

        self.blocks = []

        for idx, params in enumerate(zip(layer_params, filter_params)):
            blocks, filter_num = params

            self.blocks.append(make_basic_block_layer(filter_num=filter_num,
                                                      l2_beta=l2_beta,
                                                      blocks=blocks,
                                                      stride=2 if idx > 0 else 1))

        self.avg_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.relu)

    def get_config(self):
        return super(MyResNet, self).get_config()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.avg_pool(x)

        return self.fc(x)


def make_basic_block_layer(filter_num, blocks, l2_beta=None, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, l2_beta, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, l2_beta, stride=1))

    return res_block


def res_net_18(classes, l2_beta, blocks=(2, 2, 2, 2), filters=(64, 128, 256, 512)):
    return MyResNet(blocks, filters, classes, l2_beta)


def res_net_34(classes, l2_beta):
    return MyResNet([3, 4, 6, 3], [64, 128, 256, 512], classes, l2_beta)
