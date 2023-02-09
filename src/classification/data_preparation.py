import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import re
import random
import numpy as np


def get_label(folder):
    return int(re.findall(r'\d+', folder)[0]) - 1


def get_data(path):
    data = []
    labels = []
    for folder in os.listdir(path):
        for file in os.listdir(path + '/' + folder):
            file_path = path + '/' + folder + '/' + file
            image = load_img(file_path, target_size=(150, 150))
            image = img_to_array(image)
            image = tf.cast(image, tf.float32) / 255.0
            # image = preprocess_input(image)
            data.append(image)
            labels.append(get_label(folder))

    labels = to_categorical(labels, num_classes=8)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    return data, labels
