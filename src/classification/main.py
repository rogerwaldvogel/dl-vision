import data_preparation
import data_augmentation
from model.ModelTexture import ModelTexture
from model.custom_model import CustomModel
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50V2

if __name__ == '__main__':
    INIT_LR = 1e-4  # learning rate
    EPOCHS = 20  # number of epochs
    BS = 128  # batch size

    data, labels = data_preparation.get_data('data/Kather_texture_2016_image_tiles_5000')

    data, labels = data_augmentation.add_augmented_data(data, labels,
                                                        batch_size=BS,
                                                        number_of_batches=150)

    (train_x, val_x, train_y, val_y) = train_test_split(data, labels,
                                                        test_size=0.20, stratify=labels, random_state=42)
    (train_x, test_x, train_y, test_y) = train_test_split(train_x, train_y,
                                                          test_size=0.20, stratify=train_y, random_state=42)

    base_model = MobileNetV2(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(150, 150, 3)),
                             input_shape=(150, 150, 3))
    model = ModelTexture(base_model, "mobilenetv2")
    model.configure_model(learning_rate=INIT_LR)
    model.train(train_x, train_y, val_x, val_y, epochs=EPOCHS, batch_size=BS)

    # base_model = ResNet50V2(weights="imagenet", include_top=False,
    #                         input_tensor=Input(shape=(150, 150, 3)),
    #                         input_shape=(150, 150, 3))
    # model = ModelTexture(base_model, "resnet50v2")
    # model.configure_model(learning_rate=INIT_LR)
    # model.train(aug_per_epoch, train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)

    # model = CustomModel(None, "custom_model")
    # model.configure_model(learning_rate=INIT_LR)
    # model.train(aug_per_epoch, train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)

    model.evaluate(test_x, test_y)
