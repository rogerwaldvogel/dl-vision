from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def add_augmented_data(x, y, batch_size=32, number_of_batches=10):
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    data_generator = aug.flow(x, y, batch_size=batch_size)
    for i in range(0, number_of_batches):
        x_batch, y_batch = data_generator.next()
        x = np.concatenate((x, x_batch))
        y = np.concatenate((y, y_batch))
    return x, y
