import tensorflow as tf


def compile_model(model, learning_rate=0.001, decay_steps=50, decay_rate=0.96, beta_1=0.85, beta_2=0.995):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=beta_1, beta_2=beta_2)
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
    loss = tf.keras.losses.CategoricalCrossentropy()

    # Finally compile the model!
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.metrics.Recall(),
            tf.metrics.Precision(),
            tf.metrics.AUC(),
            tf.metrics.CategoricalAccuracy()
        ],
    )

    return model, optimizer
