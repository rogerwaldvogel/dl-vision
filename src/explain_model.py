import shap
import keras
import clean_data
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Load model
model = keras.models.load_model("texture_classification")
# Get all the training images
data = clean_data.get_data("data/Kather_texture_2016_image_tiles_5000")
# Keep a fracture
images = data[1][0]
class_names = data[0]

# Create shap plot
shap.initjs()
masker = shap.maskers.Image("inpaint_telea", images[0].shape)
explainer = shap.Explainer(model, masker, output_names=np.array(class_names))
shap_values = explainer(np.array(images[:4]), outputs=shap.Explanation.argsort.flip[:8])
shap.image_plot(shap_values)
