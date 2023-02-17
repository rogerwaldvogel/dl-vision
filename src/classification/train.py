import data_preparation
from model.model_histopathologic import ModelHistopathologic
from model.custom_model import CustomModel

from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50V2

if __name__ == '__main__':
    INIT_LR = 1e-4  # learning rate
    EPOCHS = 30  # number of epochs
    BS = 128  # batch size

    train_x, train_y, val_x, val_y, test_x, test_y = data_preparation.load_data()

    # MobileNetV2 training
    base_model = MobileNetV2(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(150, 150, 3)),
                             input_shape=(150, 150, 3))
    model = ModelHistopathologic(base_model, "mobilenetv2")
    model.configure_model(learning_rate=INIT_LR)
    model.train(train_x, train_y, val_x, val_y, epochs=EPOCHS, batch_size=BS)
    model.evaluate(test_x, test_y)

    # ResNet50V2 training
    base_model = ResNet50V2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(150, 150, 3)),
                            input_shape=(150, 150, 3))
    model = ModelHistopathologic(base_model, "resnet50v2")
    model.configure_model(learning_rate=INIT_LR)
    model.train(train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)
    model.evaluate(test_x, test_y)

    # Custom CNN model training
    model = CustomModel(None, "custom_model")
    model.configure_model(learning_rate=INIT_LR)
    model.train(train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)
    model.evaluate(test_x, test_y)
