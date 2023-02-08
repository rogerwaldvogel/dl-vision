import data_preparation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    INIT_LR = 1e-4  # learning rate
    EPOCHS = 10  # 10 are more than enough to get an accuracy of over 99%
    BS = 32  # Batchsize

    data, labels = data_preparation.get_data('data/Kather_texture_2016_image_tiles_5000')
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    classification_model = model.get_model()
    model.configure_model(classification_model, learning_rate=INIT_LR, weight_decay=INIT_LR / EPOCHS)

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    results = classification_model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    model.save("texture_classification", save_format="h5")
