from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from .tensorboard_image import save_image_to_tensorboard


class TensorBoardLime:
    def __init__(self, model, log_dir, val_images, val_labels, val_images_category):
        self.model = model
        self.log_dir = log_dir
        self.val_images = val_images
        self.val_labels = np.argmax(val_labels, axis=1)
        self.number_of_images_per_category = len(val_images_category[0])
        self.val_images_category = val_images_category
        self.class_names = ["Tumor", "Stroma", "Complex", "Lympho", "Debris", "Mucosa", "Adipose", "Empty"]

    def log_lime(self, epoch, logs):
        explainer = lime_image.LimeImageExplainer()
        for category_index, images in self.val_images_category.items():
            category_name = "epoch_lime_" + self.class_names[category_index].lower()
            title = self.class_names[category_index]
            self._log_category(explainer, images, epoch, category_name, title)

    def _log_category(self, explainer, images, epoch, category_name, title):
        fig, ax = plt.subplots(self.number_of_images_per_category, 3, figsize=(20, 20))
        image_index = 0
        fig.suptitle(title)
        for index, axis in enumerate(ax.flat):
            if index % 3 == 0:
                explanation = explainer.explain_instance(images[image_index].astype('double'),
                                                         self.model.predict,
                                                         top_labels=5,
                                                         hide_color=0, num_samples=1000)
                axis.imshow(images[image_index])
                image_index += 1
            elif index % 3 == 1:
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,
                                                            num_features=5,
                                                            hide_rest=False)
                axis.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            else:
                ind = explanation.top_labels[0]
                dict_heatmap = dict(explanation.local_exp[ind])
                heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
                im = axis.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
                plt.colorbar(im, ax=axis, shrink=0.7)

        save_image_to_tensorboard(fig, self.log_dir + "/lime", category_name, epoch)
