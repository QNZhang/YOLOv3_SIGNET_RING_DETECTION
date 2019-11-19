# -*- coding: utf-8 -*-
""" utils/plot_tools """

import os
import xml.etree.ElementTree as ET

import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patches

import settings
from .data import get_or_create_bndbox_dict
from .files import get_name_and_extension
from .kfb import read_roi_json


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
    # fig.savefig('example.png')


def plot_cervical_image_plus_bounding_boxes(image_name, save_to_disk=False, saving_folder='', draw_bbox=True):
    """
    Plots the KFB roi image (mini patch) along with its bounding boxes.
    Args:
        image_name: 'image-roiX.json'
        save_to_disk: True or False
        saving_folder: 'folder_name_to_save_roi.png'
        draw_bbox: True or False
    Usage:
        plot_cervical_image_plus_bounding_boxes(xmlfile, True, 'preview_rois')
    """
    assert isinstance(image_name, str) and image_name.endswith('.json')
    assert isinstance(save_to_disk, bool)
    assert isinstance(saving_folder, str)
    assert isinstance(draw_bbox, bool)
    print(image_name)

    if saving_folder and not os.path.exists(saving_folder):
        os.mkdir(saving_folder)

    name, _ = get_name_and_extension(image_name)

    roi, roi_anns = read_roi_json(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, image_name))

    root = ET.parse(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, name + ".xml")).getroot()
    if draw_bbox:
        for _object in root.findall('./object'):
            bndbox = {elem.tag: int(elem.text) for elem in _object.find('bndbox').getchildren()}

            if bndbox:
                cv.rectangle(
                    roi,
                    (bndbox['xmin'], bndbox['ymin']),
                    (bndbox['xmax'], bndbox['ymax']),
                    (0, 0, 255),
                    8
                )

    # cv.imshow('roi', roi)
    # cv.waitKey(1000)

    if save_to_disk:

        if saving_folder:
            filepath = os.path.join(saving_folder, name + ".jpeg")
        else:
            filepath = name + ".jpeg"

        cv.imwrite(filepath, roi)
        # cv.imwrite('ttt4.png', roi[
        #     pos_anns[0]['y']-roi_ann['y']:pos_anns[0]['y']-roi_ann['y']+pos_anns[0]['h'],
        #     pos_anns[0]['x']-roi_ann['x']:pos_anns[0]['x']-roi_ann['x']+pos_anns[0]['w']
        # ])


def create_X_cervical_images_plus_bounding_boxes(
        img_range=None, reading_folder=settings.SIGNET_TRAIN_POS_IMG_PATH,
        saving_folder='preview_rois', draw_bbox=True):
    """
    Creates and saves JPEG images of the first 'img_number' KFB roi image (mini patch)
    along with their bounding boxes.

    Notes:
    * If img_name is None, images from all minipatches in the reading directory will
      be created.
    * If there are errors when running this function, re-run it starting on the last
      image processed (review number of images in saving_folder). This error happens when
      draw_bbox = True, seems that it's and issue with CV2.
      Some common errors could be:
      - Process Python bus error (core dumped)
      - corrupted double-linked list
      - corrupted double-linked list (not small)
      - Process Python aborted (core dumped)
      - Process Python segmentation fault (core dumped)
    Args:
        img_range: tuple containing lower and upper bounds for the images to be created
        reading_folder: path to folder containing the minipatches
        saving_folder: folder name to save the images (str)
        draw_bbox: boolean indicanting to draw or not the bounding box
    Usage:
        # first 10 minipatches
        create_X_cervical_images_plus_bounding_boxes((0, 10))
        # all minipatches
        create_X_cervical_images_plus_bounding_boxes()
    """
    assert isinstance(img_range, tuple) or img_range is None
    assert os.path.exists(reading_folder)

    minipatches_list = tuple(filter(lambda x: x.endswith('.json'), os.listdir(reading_folder)))
    total_minipatches = len(minipatches_list)
    print(total_minipatches)
    lower = upper = None

    if img_range is None:
        lower, upper = 0, total_minipatches
    else:
        assert isinstance(img_range[0], int) and isinstance(img_range[1], int)
        assert img_range[0] < img_range[1]
        assert 0 <= img_range[0] < total_minipatches
        assert img_range[1] <= total_minipatches
        lower, upper = img_range[0], img_range[1]

    for index, xmlfile in enumerate(minipatches_list[lower:upper]):
        plot_cervical_image_plus_bounding_boxes(xmlfile, True, saving_folder, draw_bbox)
