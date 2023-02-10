import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from PIL import Image
import tensorflow as tf
from matplotlib import gridspec
import math


model = keras.models.load_model("texture_classification")
name_dict = {0: "TUMOR",
             1: "STROMA",
             2: "COMPLEX",
             3: "LYMPHO",
             4: "DEBRIS",
             5: "MUCOSA",
             6: "ADIPOSE"}


def load_images(path, raw: bool):
    """
    Function load the large images from the data folder and (if raw = False) normalizes it
    :param path: Path to large images
    :param raw: Bool if the function should return raw image or not
    :return: List with the normalized images
    """
    imgs = os.listdir(path)
    images = []
    if raw:
        for img in imgs:
            img_ = Image.open(path + img)
            images.append(img_)
    else:
        for img in imgs:
            img_ = Image.open(path + img)
            img_ = tf.cast(np.array(img_), tf.float32) / 255.0
            images.append(img_)

    return images


def sliding_window(width_x: int, width_y: int, classes: list, model, image):
    """
    Function applies a sliding window to a large image. For each window, the model predicts the probabilities
    for eight classes. From these eight classes, only the one specified in 'classes' are taken.
    :param width_x: Window width in x dimension
    :param width_y: Window width in y dimension
    :param classes: Which classes should be kept
    :param model: Trained model
    :param image: Test image
    :return: Array with list of probabilities
    """
    num_x = int(5000 / width_x)
    num_y = int(5000 / width_y)

    preds_list = []
    for xind in range(num_x):
        for yind in range(num_y):
            portion = image[xind * width_x: (xind + 1) * width_x, yind * width_y: (yind + 1) * width_y, :]
            img_exp = np.expand_dims(portion, axis=0)
            pred_ = model.predict(img_exp)
            preds_list.append(pred_[0, classes])

    return preds_list


def list_to_mat(pred_list: list, nwindows: int):
    """
    Function changes the type of result from list to array with matrices
    :param pred_list: List of predictions
    :param nwindows: Number of windows (used to reshape data into matrix)
    :return: Array of matrices
    """
    mat_list = []
    for cls in range(len(pred_list[0])):
        mat_list.append([val[cls] for val in pred_list[:]])
        mat_list[cls] = np.array(mat_list[cls]).reshape((nwindows, nwindows))
    return mat_list


def threshold(mat: list, thrsh: float):
    """
    Function sets every value from predicted matrix to 0 if it's lower than the threshold
    :param mat: Array with the prediction results per class
    :param thrsh: Threshold
    :return: Prediction matrix
    """
    _ = []
    for mats in mat:
        mats[mats < thrsh] = 0
        _.append(mats)
    return _


def do_plot(ax, mat: list, N: int, t_class: int, n_dict: dict, image):
    """
    Function plots on a specific ax element. Used for the subplots
    :param ax: ax subplot element
    :param mat: Array with the prediction results per class
    :param N: Which class should be plotted
    :param t_class: True class, equivalent to the correct name
    :param n_dict: Name dict for the title
    :param image: Original image (same as the one for the prediction)
    :return: Suboplot object
    """
    ax.imshow(image.convert("L"), cmap='gray')
    ax.imshow(mat[N], cmap='YlOrRd', alpha=0.3, extent=(0, 5000, 5000, 0))
    ax.title.set_text(n_dict[t_class])


def visualize(mat: list, classes: list, image):
    """
    Function plots the different predicted spots on the original image next to the original image
    :param mat: Array with the prediction results per class
    :param classes: Which classes do need to be visualized
    :param image: Original image (same as the one for the prediction)
    :return: Plot
    """
    cols = 1 if len(classes) in [1, 3] else 2
    rows = int(len(classes)/2) if len(classes) % 2 == 0 else int(math.ceil(len(classes)/2))+1
    gs = gridspec.GridSpec(rows, cols+1)
    fig = plt.figure()
    n = 0
    for col in range(cols):
        for row in range(rows):
            ax = fig.add_subplot(gs[row, col])
            do_plot(ax, mat, N=n, t_class=classes[n], n_dict=name_dict, image=image)
            n += 1
    ax2 = fig.add_subplot(gs[:, cols])
    ax2.imshow(image)
    ax2.title.set_text("RAW IMAGE")
    fig.tight_layout()
    plt.show()


classes = [0, 2, 3, 6]
imgs = load_images("data/large_images/", raw=False)
tmp = sliding_window(width_x=150, width_y=150, classes=classes, model=model, image=imgs[2])
fin = threshold(list_to_mat(pred_list=tmp, nwindows=33), 0.3)
imgs_raw = load_images("data/large_images/", raw=True)
visualize(mat=fin, classes=classes, image=imgs_raw[2])
