import tensorflow as tf

# https://github.com/calmisential/TensorFlow2.0_ResNets
def res_net(input_shape, class_names, batch_norm=None, l2_beta=None, dropout=None, output_bias=None):
    input_layer = tf.keras.layers.Input(shape=input_shape, name='InputLayer')
    reshape_layer = tf.keras.layers.Reshape(input_shape[:-1])

    core = res_net_18(
        classes=1024,
        activation='relu'
    )
    core.trainable = True

    # add regularization to layers of res net
    if l2_beta is not None:
        r = tf.keras.regularizers.l2(l2_beta)

        for layer in core.layers:
            for attr in ['kernel_regularized']:
                if hasattr(layer, attr):
                    setattr(layer, attr, r)

    model = tf.keras.models.Sequential()
    model.add(input_layer)
    model.add(reshape_layer)
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
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.down_sample = tf.keras.Sequential()
            self.down_sample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                        kernel_size=(1, 1),
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
    def __init__(self, layer_params, classes, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=classes, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avg_pool(x)

        return self.fc(x)


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def res_net_18(classes, activation):
    return MyResNet([2, 2, 2, 2], classes, activation)


def res_net_34(classes, activation):
    return MyResNet([3, 4, 6, 3], classes, activation)
