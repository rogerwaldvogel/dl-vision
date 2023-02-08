import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class MobileNetV2Texture:
    def __init__(self):
        self.model = self._get_model()

    def _get_model(self):
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))

        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(256, activation="relu")(head_model)
        head_model = Dropout(0.2)(head_model)
        head_model = Dense(8, activation="softmax")(head_model)
        model = Model(inputs=base_model.input, outputs=head_model)
        # We don't want to train any layer in the MobileNetV2 network
        for layer in base_model.layers:
            layer.trainable = False
        return model

    def configure_model(self, learning_rate=1e-4, weight_decay=1e-4):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                           metrics=["accuracy"])

    def train(self, aug, trainX, trainY, testX, testY, epochs=30, batch_size=32):
        results = self.model.fit(
            aug.flow(trainX, trainY, batch_size=batch_size),
            steps_per_epoch=len(trainX) // batch_size,
            validation_data=(testX, testY),
            validation_steps=len(testX) // batch_size,
            epochs=epochs)

        self.model.save("mobile_net_v2_classification", save_format="h5")
