from .ModelTexture import ModelTexture
import tensorflow as tf


class CustomModel(ModelTexture):
    def __init__(self, base_model, model_name):
        super().__init__(base_model, model_name)

    def _get_model(self, base_model):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((150, 150, 3)),
            tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation=tf.keras.activations.relu,
                                   kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu,
                                  kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(8, activation=tf.keras.activations.softmax)
        ], name='custom_model')
        return model
