# git checkout nicco

import os
import re
from PIL import Image
import numpy as np
import tensorflow as tf


def normalize_resize(input_image):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_image = tf.image.resize(input_image, (150, 150))
  return input_image


def get_data(path: str):
    fldrs = os.listdir(path)
    cls_names = ([re.sub(r"\d+", "", cls).lstrip("_") for cls in fldrs])
    data_array = []
    for fldr in fldrs:
        print(fldr)
        fldr_path = path + "/" + fldr
        print(fldr_path)
        files = os.listdir(fldr_path)
        tmp = []
        for file in files:
            img = Image.open(fldr_path + "/" + file)
            imarray = normalize_resize(np.array(img))
            tmp.append(imarray)
        data_array.append(tmp)
    return cls_names, data_array


test=1
