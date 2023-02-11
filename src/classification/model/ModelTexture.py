import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class ModelTexture:
    def __init__(self, base_model, model_name):
        self.model = self._get_model(base_model)
        self.model_name = model_name

    def _get_model(self, base_model):
        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(64, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.L2(0.001))(head_model)
        head_model = Dropout(0.2)(head_model)
        head_model = Dense(64, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.L2(0.001))(head_model)
        head_model = Dense(8, activation="softmax")(head_model)
        model = Model(inputs=base_model.input, outputs=head_model)
        # We don't want to train any layer of the base model
        for layer in base_model.layers:
            layer.trainable = False
        return model

    def configure_model(self, learning_rate=1e-4):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=100,
            decay_rate=0.98,
            staircase=True)

        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                           metrics=["accuracy", "Precision", "Recall"])
        print(self.model.summary())

    def train(self, train_x, train_y, val_x, val_y, epochs=30, batch_size=32):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
        result = self.model.fit(
            x=train_x,
            y=train_y,
            steps_per_epoch=len(train_x) // batch_size,
            validation_data=(val_x, val_y),
            validation_steps=len(val_x) // batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback, early_stopping_callback])

        self.model.save(f"{self.model_name}.h", save_format="h5")
        np.save(f'{self.model_name}.npy', result)

    def evaluate(self, test_x, test_y):
        print("Evaluating model...")
        self.model.evaluate(test_x, test_y)
