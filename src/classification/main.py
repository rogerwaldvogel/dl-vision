import data_preparation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.MobileNetV2Texture import MobileNetV2Texture
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    INIT_LR = 1e-4  # learning rate
    EPOCHS = 30  # number of epochs
    BS = 32  # batch size

    data, labels = data_preparation.get_data('data/Kather_texture_2016_image_tiles_5000')
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    model = MobileNetV2Texture()
    model.configure_model(learning_rate=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    model.train(aug, trainX, trainY, testX, testY, epochs=EPOCHS, batch_size=BS)


