import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from PIL import Image
import tensorflow as tf
from matplotlib import gridspec
import math


model = keras.models.load_model("models/resnet50v2.h")
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


def threshold2(mat: list, thrsh: float):
    """
    Function sets every value from predicted matrix to 0 if it's lower than the threshold
    :param mat: Array with the prediction results per class
    :param thrsh: Threshold
    :return: Prediction matrix
    """
    _ = []
    for mats in mat:
        mats[mats < thrsh] = 0
        mats[mats >= thrsh] = 1
        _.append(mats)
    return _


def do_plot(ax, mat: list, n: int, t_class: int, n_dict: dict, image):
    """
    Function plots on a specific ax element. Used for the subplots
    :param ax: ax subplot element
    :param mat: Array with the prediction results per class
    :param n: Which class should be plotted
    :param t_class: True class, equivalent to the correct name
    :param n_dict: Name dict for the title
    :param image: Original image (same as the one for the prediction)
    :return: Suboplot object
    """
    ax.imshow(image.convert("L"), cmap='gray')
    ax.imshow(mat[n], cmap='YlOrRd', alpha=0.3, extent=(0, 5000, 5000, 0))
    ax.title.set_text(n_dict[t_class])


def visualize(mat: list, cls: list, image):
    """
    Function plots the different predicted spots on the original image next to the original image
    :param mat: Array with the prediction results per class
    :param cls: Which classes do need to be visualized
    :param image: Original image (same as the one for the prediction)
    :return: Plot
    """
    cols = 1 if len(cls) in [1, 3] else 2
    rows = int(len(cls)/2) if len(cls) % 2 == 0 else int(math.ceil(len(cls)/2))+1
    gs = gridspec.GridSpec(rows, cols+1)
    fig = plt.figure()
    n = 0
    for col in range(cols):
        for row in range(rows):
            ax = fig.add_subplot(gs[row, col])
            do_plot(ax, mat, n=n, t_class=cls[n], n_dict=name_dict, image=image)
            n += 1
    ax2 = fig.add_subplot(gs[:, cols])
    ax2.imshow(image)
    ax2.title.set_text("RAW IMAGE")
    fig.tight_layout()
    plt.show()


def single_plot(mat: list, cls: int, n_dict: dict, image, true_cls: int):
    fig, ax = plt.subplots(1)
    ax.imshow(image.convert("L"), cmap='gray')
    im = ax.imshow(mat[cls], cmap='YlOrRd', alpha=0.3, extent=(0, 5000, 5000, 0))
    cb = fig.colorbar(im)
    cb.set_label(label='Probability', size=12)
    ax.title.set_text(n_dict[true_cls])
    plt.show()
    


classes = [0, 1, 2, 3]
# classes = [4, 5, 6]
img_ind = 1

imgs = load_images("data/large_images/", raw=False)
tmp = sliding_window(width_x=150, width_y=150, classes=classes, model=model, image=imgs[img_ind])
fin = threshold(list_to_mat(pred_list=tmp, nwindows=33), 0.3)
imgs_raw = load_images("data/large_images/", raw=True)
visualize(mat=fin, cls=classes, image=imgs_raw[img_ind])
## Single Plots of images overlayed
#single_plot(mat=fin, cls=0, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=0)
#single_plot(mat=fin, cls=1, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=1)
#single_plot(mat=fin, cls=2, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=2)
#single_plot(mat=fin, cls=3, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=3)
#single_plot(mat=fin, cls=0, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=4)
#single_plot(mat=fin, cls=1, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=5)
#single_plot(mat=fin, cls=2, image=imgs_raw[img_ind], n_dict=name_dict, true_cls=6)

## 1 and 0 based on pred mat
"""
temp = threshold2(list_to_mat(pred_list=tmp, nwindows=33), 0.5)
fig, ax = plt.subplots()
min_val, max_val = 0, 33
ax.matshow(temp[0], cmap=plt.cm.Blues)


tmp1 = temp[0] + temp[1]

classes = [0, 1, 2, 3, 4, 5, 6]
model_cst = keras.models.load_model("models/custom_model.h")
custom = sliding_window(width_x=150, width_y=150, classes=classes, model=model_cst, image=imgs[img_ind])
custom_mat = threshold2(list_to_mat(pred_list=custom, nwindows=33), 0.5)

model_res = keras.models.load_model("models/resnet50v2.h")
resnet = sliding_window(width_x=150, width_y=150, classes=classes, model=model_res, image=imgs[img_ind])
resnet_mat = threshold2(list_to_mat(pred_list=resnet, nwindows=33), 0.5)

model_mob = keras.models.load_model("models/mobilenetv2.h")
mobilenet = sliding_window(width_x=150, width_y=150, classes=classes, model=model_mob, image=imgs[img_ind])
mobilenet_mat = threshold2(list_to_mat(pred_list=mobilenet, nwindows=33), 0.5)

fin = []
for num in range(7):
    fin.append(custom_mat[num] + resnet_mat[num] + mobilenet_mat[num])

hover = []
for row in range(33):
    for col in range(33):
        if custom_mat[0][row, col] == resnet_mat[0][row, col] == mobilenet_mat[0][row, col] == 1:
            hover.append(["All models"])
        elif (custom_mat[0][row, col] == resnet_mat[0][row, col] == 1) & (mobilenet_mat[0][row, col] == 0):
            hover.append(["Cutsom Model and Resnet Model"])
        elif (custom_mat[0][row, col] == mobilenet_mat[0][row, col] == 1) & (resnet_mat[0][row, col] == 0):
            hover.append(["Cutsom Model and Mobilenet Model"])
        elif (resnet_mat[0][row, col] == mobilenet_mat[0][row, col] == 1) & (custom_mat[0][row, col] == 0):
            hover.append(["Resnet Model and Mobilenet Model"])
        elif (custom_mat[0][row, col] == 1) & (resnet_mat[0][row, col] == mobilenet_mat[0][row, col] == 0):
            hover.append(["Only Custom Model"])
        elif (resnet_mat[0][row, col] == 1) & (custom_mat[0][row, col] == mobilenet_mat[0][row, col] == 0):
            hover.append(["Only Resnet Model"])
        elif (mobilenet_mat[0][row, col] == 1) & (custom_mat[0][row, col] == resnet_mat[0][row, col] == 0):
            hover.append(["Only Mobilenet Model"])
        else:
            hover.append(["No Model"])




import plotly
import pandas as pd

colorscale = [[0, '#454D59'],[0.5, '#FFFFFF'], [1, '#F1C40F']]

zz = pd.DataFrame(np.vstack(list_to_mat(pred_list=hover, nwindows=33)))

x = zz.columns.tolist()
y = zz.index.tolist()
z = fin[0]


hovertext = list()
for yi, yy in enumerate(y):
    hovertext.append(list())
    for xi, xx in enumerate(x):
        hovertext[-1].append('X: {}<br />Y: {}<br />Which Models: {}'.format(xx, yy, zz[xi][yi]))

data = [plotly.graph_objs.Heatmap(z=z,
                                  colorscale=colorscale,
                                  x=x,
                                  y=y,
                                  hoverinfo='text',
                                  text=hovertext)]

layout = plotly.graph_objs.Layout(autosize=True,
                                  font=dict(family="Courier New"),
                                  width=1200,
                                  height=1000,
                                  margin=plotly.graph_objs.Margin(l=150,
                                                                  r=160,
                                                                  b=50,
                                                                  t=100,
                                                                  pad=3)
                                 )

fig = plotly.graph_objs.Figure(data=data, layout=layout)
fig['layout']['yaxis']['autorange'] = "reversed"
fig.show()

fig.update_layout(scene={'yaxis': {'autorange': 'reversed'}})
z = pd.DataFrame(np.vstack(list_to_mat(pred_list=hover, nwindows=33)))
# Setup the different axes
fig.add_trace(go.Heatmap(
    z=fin[0],
    customdata=hover,
    hovertemplate='<b>z1:%{z:.3f}</b><br>z2:%{customdata[0]:.3f} <br>z3: %{customdata[1]:.3f} ',
    coloraxis="coloraxis1", name=''),
    1, 1)


# plotly 5.13.0
# nbformat 5.7.3
import plotly.express as px
import plotly.figure_factory as ff
# fig = ff.create_annotated_heatmap(fin[0], text=hover[::-1], hoverinfo='text', annotation_text = None)
fig = px.imshow(fin[0], labels = dict(x = "X-axis", y = "Y-axis", color = "Legend"),
                x = hover,
                y = hover)
fig.show()
fig.write_html("html/file.html")



import plotly.figure_factory as ff
import numpy as np
np.random.seed(1)

z = np.random.randn(20, 20)
z_text = np.around(z, decimals=2) # Only show rounded value (full value on hover)

fig = ff.create_annotated_heatmap(fin[0], annotation_text=None, colorscale='Greys', text=hover,
                                  hoverinfo='text')

# Make text size smaller
for i in range(len(fig.layout.annotations)):
    fig.layout.annotations[i].font.size = 8

fig.show()


for i in range(33):
    for j in range(33):
        c = int(temp[0][j,i])
        ax.text(i+0.5, j+0.5, str(c), va='center', ha='center')
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_xticks(np.arange(max_val))
ax.set_yticks(np.arange(max_val))
ax.grid()

plt.matshow(temp[0], cmap=plt.cm.Blues)
"""