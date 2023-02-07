import data_preparation
import tensorflow as tf

if __name__ == '__main__':
    data, labels = data_preparation.get_data('data/Kather_texture_2016_image_tiles_5000')
    # shuffel data with for tensorflow
    # split data into train and test
    data, labels = tf.random_index_shuffle(data, labels)

    test = 1
