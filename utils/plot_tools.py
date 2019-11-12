# -*- coding: utf-8 -*-
""" utils/plot_tools """

import json
import os

import cv2 as cv
import kfbReader
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


def plot_cervical_image_plus_bounding_boxes(image_name, save_disk=False):
    """
    Plots the KFB image and draws its bouding boxes.
    """
    # TODO: Finish this after creating the mini patches????
    name, _ = get_name_and_extension(image_name)

    with open(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, name + ".json")) as jfile:
        ann = json.load(jfile)
        scale = 20
        read = kfbReader.reader()
        read.setReadScale(scale)
        read.ReadInfo(
            os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, image_name),
            scale,
            True
        )
        print(scale)
        height = read.getHeight()
        width = read.getWidth()
        scale = read.getReadScale()
        print('height: ', height)
        print('width: ', width)
        print('scale: ', scale)

        roi_ann = ann[0]
        pos_anns = ann[1:]

        roi = read.ReadRoi(roi_ann['x'], roi_ann['y'], roi_ann['w'], roi_ann['h'], scale=20)

        for pos_ann in pos_anns:
            cv.rectangle(
                roi,
                (pos_ann['x']-roi_ann['x'], pos_ann['y']-roi_ann['y']),
                (pos_ann['x']-roi_ann['x']+pos_ann['w'], pos_ann['y']-roi_ann['y']+pos_ann['h']),
                (0, 0, 255),
                8
            )

        # cv.imshow('roi', roi)
        # cv.waitKey(1000)
        # cv.imwrite("ttt2.png", roi)

        cv.imwrite('ttt4.png', roi[
            pos_anns[0]['y']-roi_ann['y']:pos_anns[0]['y']-roi_ann['y']+pos_anns[0]['h'],
            pos_anns[0]['x']-roi_ann['x']:pos_anns[0]['x']-roi_ann['x']+pos_anns[0]['w']
        ])

        cv.imwrite('ttt3.png', read.ReadRoi(
            pos_anns[0]['x'], pos_anns[0]['y'],
            pos_anns[0]['w'], pos_anns[0]['h'], scale=20))
