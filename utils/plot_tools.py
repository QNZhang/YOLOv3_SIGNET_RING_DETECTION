# -*- coding: utf-8 -*-
""" utils/plot_tools """

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches

import settings
from .data import get_or_create_bndbox_dict
from .files import get_name_and_extension


def plot_img_plus_bounding_boxes(image_name, fig_width=15, fig_height=13):
    """
    img_name: <name>.<extension>
    Plots the image along with its bounding boxes
    """
    assert isinstance(image_name, str) and isinstance(fig_width, int) and \
        isinstance(fig_height, int)

    name, _ = get_name_and_extension(image_name)
    bndbox_dictionary = get_or_create_bndbox_dict()
    img = mpimg.imread(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, image_name))
    img_bndboxes = bndbox_dictionary[name]

    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))

    for bndbox in img_bndboxes:
        rect = patches.Rectangle(
            (float(bndbox.bndbox.xmin), float(bndbox.bndbox.ymin)),
            float(bndbox.bndbox.xmax) - float(bndbox.bndbox.xmin),
            float(bndbox.bndbox.ymax) - float(bndbox.bndbox.ymin),
            linewidth=2, edgecolor='b', facecolor='none'
        )
        ax.add_patch(rect)

    ax.imshow(img)
    plt.show()
