from PIL import Image
import tensorflow as tf
import os

files = os.listdir("data/large_images/")
for file in files:
    img = Image.open("data/large_images/" + file)
    input_image = tf.cast(np.array(img), tf.float32) / 255.0
    print(input_image.shape)