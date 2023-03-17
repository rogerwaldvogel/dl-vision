import sliding_window
import numpy as np
import keras
import plotly
import pandas as pd

# Define classes and large images
classes = [0, 1, 2, 3, 4, 5, 6]
img_ind = 1
imgs = sliding_window.load_images("data/large_images/", raw=False)

# Load models
model_cst = keras.models.load_model("models/custom_model.h")
custom = sliding_window.sliding_window(width_x=150, width_y=150, classes=classes, model=model_cst, image=imgs[img_ind])
custom_mat = sliding_window.threshold2(sliding_window.list_to_mat(pred_list=custom, nwindows=33), 0.5)

model_res = keras.models.load_model("models/resnet50v2.h")
resnet = sliding_window.sliding_window(width_x=150, width_y=150, classes=classes, model=model_res, image=imgs[img_ind])
resnet_mat = sliding_window.threshold2(sliding_window.list_to_mat(pred_list=resnet, nwindows=33), 0.5)

model_mob = keras.models.load_model("models/mobilenetv2.h")
mobilenet = sliding_window.sliding_window(width_x=150, width_y=150, classes=classes, model=model_mob, image=imgs[img_ind])
mobilenet_mat = sliding_window.threshold2(sliding_window.list_to_mat(pred_list=mobilenet, nwindows=33), 0.5)
print("Loaded all models")
print("Generating Hover Matrix")


def add_all():
    """
    Add all model outputs together
    """
    fin = []
    for num in range(7):
        fin.append(custom_mat[num] + resnet_mat[num] + mobilenet_mat[num])
    return fin


def add_res_mob():
    """
    Add resnet and mobilenet model outputs together
    """
    fin = []
    for num in range(7):
        fin.append(resnet_mat[num] + mobilenet_mat[num])
    return fin


def add_res_cust():
    """
    Add custom and resnet model outputs together
    """
    fin = []
    for num in range(7):
        fin.append(custom_mat[num] + resnet_mat[num])
    return fin


def add_mob_cust():
    """
    Add custom and mobilenet model outputs together
    """
    fin = []
    for num in range(7):
        fin.append(custom_mat[num] + mobilenet_mat[num])
    return fin


def hover_all(ind: int):
    """
    Detect which model (of all models) predicts true and which false.
    Depending on the result, write the text into a list
    :param ind: Which category should be used (cancer, adipose etc.)
    :return: list
    """
    hover_all = []
    for row in range(33):
        for col in range(33):
            if custom_mat[ind][row, col] == resnet_mat[ind][row, col] == mobilenet_mat[ind][row, col] == 1:
                hover_all.append(["All models"])
            elif (custom_mat[ind][row, col] == resnet_mat[ind][row, col] == 1) & (mobilenet_mat[ind][row, col] == 0):
                hover_all.append(["Custom Model and Resnet Model"])
            elif (custom_mat[ind][row, col] == mobilenet_mat[ind][row, col] == 1) & (resnet_mat[ind][row, col] == 0):
                hover_all.append(["Custom Model and Mobilenet Model"])
            elif (resnet_mat[ind][row, col] == mobilenet_mat[ind][row, col] == 1) & (custom_mat[ind][row, col] == 0):
                hover_all.append(["Resnet Model and Mobilenet Model"])
            elif (custom_mat[ind][row, col] == 1) & (resnet_mat[ind][row, col] == mobilenet_mat[ind][row, col] == 0):
                hover_all.append(["Only Custom Model"])
            elif (resnet_mat[ind][row, col] == 1) & (custom_mat[ind][row, col] == mobilenet_mat[ind][row, col] == 0):
                hover_all.append(["Only Resnet Model"])
            elif (mobilenet_mat[ind][row, col] == 1) & (custom_mat[ind][row, col] == resnet_mat[ind][row, col] == 0):
                hover_all.append(["Only Mobilenet Model"])
            else:
                hover_all.append(["No Model"])

    return hover_all


def hover_res_mob(ind: int):
    """
    Detect which model (resnet and mobilenet) predicts true and which false.
    Depending on the result, write the text into a list
    :param ind: Which category should be used (cancer, adipose etc.)
    :return: list
    """
    hover_mob_res = []
    for row in range(33):
        for col in range(33):
            if resnet_mat[ind][row, col] == mobilenet_mat[ind][row, col] == 1:
                hover_mob_res.append(["Both models"])
            elif (resnet_mat[ind][row, col] == 1) & (mobilenet_mat[ind][row, col] == 0):
                hover_mob_res.append(["Only Resnet Model"])
            elif (resnet_mat[ind][row, col] == 0) & (mobilenet_mat[ind][row, col] == 1):
                hover_mob_res.append(["Only Mobilenet Model"])
            else:
                hover_mob_res.append(["No Model"])
    return hover_mob_res


def hover_res_cust(ind: int):
    """
    Detect which model (resnet and mobilenet) predicts true and which false.
    Depending on the result, write the text into a list
    :param ind: Which category should be used (cancer, adipose etc.)
    :return: list
    """
    hover_cust_res = []
    for row in range(33):
        for col in range(33):
            if resnet_mat[ind][row, col] == custom_mat[ind][row, col] == 1:
                hover_cust_res.append(["Both models"])
            elif (resnet_mat[ind][row, col] == 1) & (custom_mat[ind][row, col] == 0):
                hover_cust_res.append(["Only Resnet Model"])
            elif (resnet_mat[ind][row, col] == 0) & (custom_mat[ind][row, col] == 1):
                hover_cust_res.append(["Only Custom Model"])
            else:
                hover_cust_res.append(["No Model"])
    return hover_cust_res


def hover_mob_cust(ind: int):
    """
    Detect which model (custom and mobilenet) predicts true and which false.
    Depending on the result, write the text into a list
    :param ind: Which category should be used (cancer, adipose etc.)
    :return: list
    """
    hover_cust_mob = []
    for row in range(33):
        for col in range(33):
            if mobilenet_mat[ind][row, col] == custom_mat[ind][row, col] == 1:
                hover_cust_mob.append(["Both models"])
            elif (mobilenet_mat[ind][row, col] == 1) & (custom_mat[ind][row, col] == 0):
                hover_cust_mob.append(["Only Mobilenet Model"])
            elif (mobilenet_mat[ind][row, col] == 0) & (custom_mat[ind][row, col] == 1):
                hover_cust_mob.append(["Only Custom Model"])
            else:
                hover_cust_mob.append(["No Model"])
    return hover_cust_mob


def hover_plt(hover_text: list, tot_scores, ind: int, save: bool, name: None):
    """
    Create the plotly graphic to hover.
    :param hover_text: list from the model hover text generation
    :param tot_scores: Give the total scores from the model addition
    :param ind: Which category should be used (cancer, adipose etc.)
    :param save: Should the resulting graphic be saved
    :param name: If save, what name should the object have
    :return: html graphic
    """
    zz = pd.DataFrame(np.vstack(sliding_window.list_to_mat(pred_list=hover_text, nwindows=33)))

    x = zz.columns.tolist()
    y = zz.index.tolist()
    z = tot_scores[ind]

    hovertext = list()
    for yi, yy in enumerate(y):
        hovertext.append(list())
        for xi, xx in enumerate(x):
            hovertext[-1].append('X: {}<br />Y: {}<br />Which Models: {}'.format(xx, yy, zz[xi][yi]))
    data = [plotly.graph_objs.Heatmap(z=z,
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
    if save and name is not None:
        fig.write_html("html/" + name + ".html")


hover_plt(hover_text=hover_all(ind=0), tot_scores=add_all(), ind=0, save=True, name="all_models")
hover_plt(hover_text=hover_res_mob(ind=0), tot_scores=add_res_mob(), ind=0, save=True, name="resnet_mobile")
hover_plt(hover_text=hover_mob_cust(ind=0), tot_scores=add_mob_cust(), ind=0, save=True, name="mobile_custom")
hover_plt(hover_text=hover_res_cust(ind=0), tot_scores=add_res_cust(), ind=0, save=True, name="resnet_custom")
