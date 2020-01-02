"""
Utilities for generating tensorboard plots for
GAN architectures

Peter J. Thomas
02 Jan 2020 (Happy New Years!)
"""

import maplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_images(real_image,
                gen_image,
                y_real,
                y_gen,
                pred_tags,
                true_tags,
                pred_tags_gen,
                true_tags_gen,
                tag_list):
    """
    Util to plot a real image next to a generated image, and
    to display the discriminator's prediction on whether the
    imagery was a forgery or not
    """
    # plot generated and real image
    fig = plt.figure(figsize=(12, 24))

    # plot real image
    ax_real = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    matplotlib_imshow(real_image)

    plot_text(ax_real, y_real, pred_tags,
              true_tags, tag_list, real_image=True)

    # plot generated image
    ax_gen = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    matplotlib_imshow(gen_image)

    plot_text(ax_gen, y_gen, pred_tags_gen,
              true_tags_gen, tag_list, real_image=False)

    return fig

def plot_text(subplot,
              y,
              pred_tags,
              true_tags,
              tag_list,
              real_image=True):

    """
    Displays text detailing tags that the discriminator predicted as well as the
    truth tags for the image below the given matplotlib image subplot
    """
    # convert forgery labels and tag predictions into numpy arrays
    y_np = y.eval()
    np_pred_tags = pred_tags.eval()
    np_true_tags = true_tags.eval()

    # Display discriminator's forgery score over image
    if real_image:
        title_text = "real_image"
    else:
        title_text = "generated_image"

    subplot.set_title("{0}, {1:.1f}%\n(tags: {1}".format(
        title_text,
        y_np * 100.0,
    ), color=("green" if y_np > 0.50 else "red"))

    # filter for predicted tags with confidences > 50%
    pred_tag_indices = np.argwhere(np_pred_tags >= 0.50)

    # Retrieve indices of true tags in tag list
    # NOTE: user threshold of 0.90 due to possible label smoothing
    true_tag_indices = np.argwhere(np_true_tags >= 0.90)

    # Define bounds for labels below image:
    bounds = (.50 - 0.03 * (len(pred_tag_indices))/2, .50 + 0.03 * (len(pred_tag_indices))/2)

    subplot.text(bounds[0] - 0.03, 0.05,
            "predicted_tags: ", ha="center", va="bottom", size="medium")

    # Display discriminator's predictions for image tags below image
    for pred_tag_idx, counter in enumerate(pred_tag_indices):
        subplot.text(bounds[0] + (counter * 0.03), 0.05,
                tag_list[pred_tag_idx], ha="center", va="bottom", size="medium",
                color=("green" if pred_tag_idx in true_tag_indices else "red")
                )

    # Now do the same for the true tags for the image
    bounds = (.50 - 0.03 * (len(true_tag_indices))/2, .50 + 0.03 * (len(true_tag_indices))/2)

    subplot.text(bounds[0] - 0.03, 0.2,
            "truth_tags: ", ha="center", va="bottom", size="medium")

    for true_tag_idx, counter in enumerate(true_tag_indices):
        subplot.text(bounds[0] + (counter * 0.03), 0.02,
                tag_list[true_tag_idx], ha="center", va="bottom", size="medium"
                )

def matplotlib_imshow(image, one_channel=False):
    """
    Copied from pytorch tensorboard tutorial
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    """
    # Evaluate image tensor to convert to numpy array
    np_image = image.eval()
    if one_channel:
        np_image = np.mean(image, axis=2)

    np_image = np_image / 2

    if one_channel:
        plt.imshow(np_image, cmap='Greys')

    else:
        plt.imshow(np_image)




