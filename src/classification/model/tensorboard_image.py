import tensorflow as tf
import io
import matplotlib.pyplot as plt


def save_image_to_tensorboard(figure, log_dir, summary_description, epoch):
    image = _plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.image(summary_description, image, step=epoch)


def get_images_per_category(val_images, val_labels, number_of_images_per_category):
    category = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for index, val_image in enumerate(val_images):
        label = val_labels[index]
        if len(category[label]) < number_of_images_per_category:
            category[label].append(val_image)
        if not _check_proceed(category.values(), number_of_images_per_category):
            break

    return category


def _check_proceed(items, number_of_images_per_category):
    for items in items:
        if len(items) < number_of_images_per_category:
            return True
    return False


def _plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
