from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(150, 150, 3)))

    head_model = base_model.output
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(8, activation="softmax")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)
    # We don't want to train any layer in the MobileNetV2 network
    for layer in base_model.layers:
        layer.trainable = False
    return model


def configure_model(model, learning_rate=1e-4, weight_decay=1e-4):
    opt = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    return model
