import shap
import keras
import clean_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

model = keras.models.load_model("texture_classification")
data = clean_data.get_data("data/Kather_texture_2016_image_tiles_5000")
images = data[1][0]
class_names = data[0]


shap.initjs()
masker = shap.maskers.Image("inpaint_telea", images[0].shape)
explainer = shap.Explainer(model, masker, output_names=np.array(class_names))
shap_values = explainer(np.array(images[:4]), outputs=shap.Explanation.argsort.flip[:8])
shap.image_plot(shap_values)
