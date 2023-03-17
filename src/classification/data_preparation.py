import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pickle

import src.classification.data_augmentation as data_augmentation


def load_data():
    with open('data/pickle/train_x.pickle', 'rb') as f:
        train_x = pickle.load(f)
    with open('data/pickle/train_y.pickle', 'rb') as f:
        train_y = pickle.load(f)

    with open('data/pickle/val_x.pickle', 'rb') as f:
        val_x = pickle.load(f)
    with open('data/pickle/val_y.pickle', 'rb') as f:
        val_y = pickle.load(f)

    with open('data/pickle/test_x.pickle', 'rb') as f:
        test_x = pickle.load(f)
    with open('data/pickle/test_y.pickle', 'rb') as f:
        test_y = pickle.load(f)

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_label(folder):
    return int(re.findall(r'\d+', folder)[0]) - 1


def get_data(path, convert_to_categorical=True, target_size=(150, 150)):
    data = []
    labels = []
    for folder in os.listdir(path):
        for file in os.listdir(path + '/' + folder):
            file_path = path + '/' + folder + '/' + file
            image = load_img(file_path, target_size=target_size)
            image = img_to_array(image)
            image = tf.cast(image, tf.float32) / 255.0
            data.append(image)
            labels.append(get_label(folder))

    if convert_to_categorical:
        labels = to_categorical(labels, num_classes=8)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    return data, labels


def prepare_data():
    data, labels = get_data('data/Kather_texture_2016_image_tiles_5000')

    data, labels = data_augmentation.add_augmented_data(data, labels,
                                                        batch_size=128,
                                                        number_of_batches=150)
    (train_x, val_x, train_y, val_y) = train_test_split(data, labels,
                                                        test_size=0.20, stratify=labels, random_state=42)
    (train_x, test_x, train_y, test_y) = train_test_split(train_x, train_y,
                                                          test_size=0.20, stratify=train_y, random_state=42)

    with open('data/pickle/train_x.pickle', 'wb') as f:
        pickle.dump(train_x, f)
    with open('data/pickle/train_y.pickle', 'wb') as f:
        pickle.dump(train_y, f)

    with open('data/pickle/val_x.pickle', 'wb') as f:
        pickle.dump(val_x, f)
    with open('data/pickle/val_y.pickle', 'wb') as f:
        pickle.dump(val_y, f)

    with open('data/pickle/test_x.pickle', 'wb') as f:
        pickle.dump(test_x, f)
    with open('data/pickle/test_y.pickle', 'wb') as f:
        pickle.dump(test_y, f)


if __name__ == '__main__':
    prepare_data()
