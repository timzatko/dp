import tensorflow as tf


# https://github.com/calmisential/TensorFlow2.0_ResNet
def res_net(input_shape,
            class_names,
            is_3D=False,
            l2_beta=None,
            batch_norm=False,
            dropout=None,
            output_bias=None,
            blocks=(2, 2, 2, 2),
            filters=(64, 128, 256, 512)):
            
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape, name='InputLayer'))
    
    if not is_3D:
        model.add(tf.keras.layers.Reshape(input_shape[:-1]))
        
    model.add(res_net_18(
        classes=512,
        activation='relu',
        l2_beta=l2_beta,
        batch_norm=batch_norm,
        blocks=blocks,
        filters=filters,
        is_3D=is_3D,
    ))
    
    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)

    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2))

    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    l2 = None
    if l2_beta is not None:
        l2 = tf.keras.regularizers.l2(l=l2_beta)

    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2))

    # Dropout
    if dropout is not None:
        model.add(tf.keras.layers.Dropout(dropout))

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax', bias_initializer=output_bias))

    return model


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, is_3D, filter_num, l2_beta, batch_norm, stride=1):
        super(BasicBlock, self).__init__()
        
        self.batch_norm = batch_norm
        
        if not is_3D:
            self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(3, 3),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_beta) if l2_beta is not None else None,
                                                strides=stride,
                                                padding="same")
        else:
            self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                                kernel_size=(3, 3, 3),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_beta) if l2_beta is not None else None,
                                                strides=stride,
                                                padding="same")
        if self.batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
        
        if not is_3D:
            self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                                kernel_size=(3, 3),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_beta) if l2_beta is not None else None,
                                                strides=1,
                                                padding="same")
        else:
            self.conv2 = tf.keras.layers.Conv3D(filters=filter_num,
                                                kernel_size=(3, 3, 3),
                                                kernel_regularizer=tf.keras.regularizers.l2(l2_beta) if l2_beta is not None else None,
                                                strides=1,
                                                padding="same")
        if self.batch_norm:
            self.bn2 = tf.keras.layers.BatchNormalization()
        
        if stride != 1:
            self.down_sample = tf.keras.Sequential()
            
            if not is_3D:
                self.down_sample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                            kernel_size=(1, 1),
                                                            kernel_regularizer=tf.keras.regularizers.l2(l2_beta) if l2_beta is not None else None,
                                                            strides=stride))
            else:
                self.down_sample.add(tf.keras.layers.Conv3D(filters=filter_num,
                                                            kernel_size=(1, 1, 1),
                                                            kernel_regularizer=tf.keras.regularizers.l2(l2_beta) if l2_beta is not None else None,
                                                            strides=stride))
                
            self.down_sample.add(tf.keras.layers.BatchNormalization())
        else:
            self.down_sample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.down_sample(inputs)

        x = self.conv1(inputs)
        
        if self.batch_norm:
            x = self.bn1(x, training=training)
        
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        if self.batch_norm:
            x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class MyResNet(tf.keras.Model):
    def __init__(self, blocks, filters, classes, activation, l2_beta=None, batch_norm=None, is_3D=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.batch_norm = batch_norm
        
        l2 = None
        if l2_beta is not None:
            l2 = tf.keras.regularizers.l2(l=l2_beta)

        if not is_3D:
            self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(7, 7),
                                                kernel_regularizer=l2,
                                                strides=2,
                                                padding="same")
        else:
            self.conv1 = tf.keras.layers.Conv3D(filters=64,
                                                kernel_size=(7, 7, 7),
                                                kernel_regularizer=l2,
                                                strides=2,
                                                padding="same")
 
        if self.batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
    
        if not is_3D:
            self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                   strides=2,
                                                   padding="same")
        else:
            self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3),
                                                   strides=2,
                                                   padding="same")

        self.blocks = []

        for idx, params in enumerate(zip(blocks, filters)):
            block_num, filter_num = params

            self.blocks.append(make_basic_block_layer(is_3D,
                filter_num=filter_num,
                l2_beta=l2_beta,
                batch_norm=batch_norm,
                block_num=block_num,
                stride=2 if idx > 0 else 1))

        if not is_3D:
            self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        else:
             self.avg_pool = tf.keras.layers.GlobalAveragePooling3D()
                
        self.fc = tf.keras.layers.Dense(units=classes, activation=activation)

    def get_config(self):
        return super(MyResNet, self).get_config()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        
        if self.batch_norm:
            x = self.bn1(x, training=training)
            
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        for block in self.blocks:
            x = block(x, training=training)

        x = self.avg_pool(x)

        return self.fc(x)


def make_basic_block_layer(is_3D, filter_num, block_num, l2_beta, batch_norm, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(is_3D, filter_num, stride=stride, l2_beta=l2_beta, batch_norm=batch_norm))

    for _ in range(1, block_num):
        res_block.add(BasicBlock(is_3D, filter_num, stride=1, l2_beta=l2_beta, batch_norm=batch_norm))

    return res_block


def res_net_18(is_3D, classes, activation, batch_norm=False, l2_beta=None, blocks=(2, 2, 2, 2), filters=(64, 128, 256, 512)):
    return MyResNet(blocks, filters, classes, activation, batch_norm=batch_norm, l2_beta=l2_beta, is_3D=is_3D)


def res_net_34(is_3D, classes, activation, batch_norm=False, l2_beta=None, blocks=(3, 4, 6, 3), filters=(64, 128, 256, 512)):
    return MyResNet(blocks, filters, classes, activation, batch_norm=batch_norm, l2_beta=l2_beta, is_3D=is_3D)
