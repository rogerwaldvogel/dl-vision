import sklearn
import itertools
import numpy as np
import matplotlib.pyplot as plt

from .tensorboard_image import save_image_to_tensorboard


class TensorBoardCM:

    def __init__(self, model, log_dir, val_images, val_labels):
        self.model = model
        self.log_dir = log_dir
        self.val_images = val_images
        self.val_labels = np.argmax(val_labels, axis=1)
        self.class_names = ["Tumor", "Stroma", "Complex", "Lympho", "Debris", "Mucosa", "Adipose", "Empty"]

    def log_confusion_matrix(self, epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.val_images)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.val_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        save_image_to_tensorboard(figure, self.log_dir + '/cm', "epoch_confusion_matrix", epoch)

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
