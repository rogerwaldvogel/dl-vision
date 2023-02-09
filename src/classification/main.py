import data_preparation
import data_augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.ModelTexture import ModelTexture
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionResNetV2


if __name__ == '__main__':
    INIT_LR = 1e-4  # learning rate
    EPOCHS = 50  # number of epochs
    BS = 32  # batch size

    data, labels = data_preparation.get_data('data/Kather_texture_2016_image_tiles_5000')
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                          test_size=0.20, stratify=labels, random_state=42)
    train_x, train_y = data_augmentation.add_augmented_data(train_x, train_y,
                                                            batch_size=BS,
                                                            number_of_batches=100)
    aug_per_epoch = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
    # model = ModelTexture(base_model, "mobilenetv2")
    # model.configure_model(learning_rate=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    # model.train(aug_per_epoch, train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)

    # base_model = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
    # model = ModelTexture(base_model, "xception")
    # model.configure_model(learning_rate=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    # model.train(aug_per_epoch, train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)

    base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
    model = ModelTexture(base_model, "inceptionresnetv2")
    model.configure_model(learning_rate=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    model.train(aug_per_epoch, train_x, train_y, test_x, test_y, epochs=EPOCHS, batch_size=BS)
