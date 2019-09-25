# -*- coding: utf-8 -*-
""" challenge utils  """

from copy import deepcopy
import os
import shutil

import cv2
import numpy as np
import torch
import xmltodict
from PIL import Image, ImageDraw

from . import settings


def initial_validation_cleaning():
    """ Verifies the input folder exists and cleans the output folder """
    if not os.path.exists(settings.INPUT_FOLDER):
        raise FileNotFoundError(
            "You must create a folder called {} and put the images to be evaluated there."
            .format(settings.INPUT_FOLDER))

    # Cleaning output folder
    if os.path.exists(settings.OUTPUT_FOLDER):
        shutil.rmtree(settings.OUTPUT_FOLDER)
    os.makedirs(settings.OUTPUT_FOLDER)


def use_cuda():
    """ Returns True if cuda is availabla and has been enabled in the configuration file """
    return torch.cuda.is_available() and settings.USE_CUDA


def evaluation(x, y, cut_size, w, h, fimg, model):
    """  """
    fimg = cv2.imread(fimg.filename)
    image = fimg[y:y+cut_size, x:x+cut_size]
    # cv2.imshow("cropped", image)

    #######
    results = model.get_predictions(image=image, plot=False)

    if len(results) == 0:
        return None

    c = results.cpu().numpy()
    #######
    # c = coco_demo.select_top_predictions(c)
    if(x != 0 and y != 0 and x+cut_size != w and y+cut_size != h):
        i = 0
        while(i < c.shape[0]):
            if(c[i, 0] < settings.BOARDCACHE or c[i, 1] < settings.BOARDCACHE or c[i, 2] < settings.BOARDCACHE or c[i, 3] < settings.BOARDCACHE):
                c = np.delete(c, i, axis=0)
                i -= 1
            i += 1
    i = 0

    while(i < c.shape[0]):
        c[i, 0] += x
        c[i, 2] += x
        c[i, 1] += y
        c[i, 3] += y
        i += 1

    return c


def process_input_files(model, draw_annotations=False):
    """  """
    for fileimg in tuple(filter(lambda x: x.endswith('.jpeg'), os.listdir(settings.INPUT_FOLDER))):
        print(fileimg)
        predictions = [[0, 0, 0, 0, 0]]
        fimg = Image.open(os.path.join(settings.INPUT_FOLDER, fileimg))
        w, h = fimg.size
        y = 0

        while(y <= (h-settings.CUT_SIZE)):
            x = 0

            while(x <= (w-settings.CUT_SIZE)):
                eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, fimg, model)
                if eval_results is not None:
                    predictions = np.vstack((predictions, eval_results))
                x = x+settings.OVERLAP
                # print(x)

            x = w - settings.CUT_SIZE
            eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, fimg, model)
            if eval_results is not None:
                predictions = np.vstack((predictions, eval_results))
            y = y+settings.OVERLAP
            # print(y)

        x = 0
        y = h - settings.CUT_SIZE

        while(x <= (w-settings.CUT_SIZE)):
            eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, fimg, model)
            if eval_results is not None:
                predictions = np.vstack((predictions, eval_results))
            x = x+settings.OVERLAP
            # print(x)

        eval_results = evaluation(
            w-settings.CUT_SIZE, h-settings.CUT_SIZE, settings.CUT_SIZE, w, h, fimg, model)
        if eval_results is not None:
            predictions = np.vstack((predictions, eval_results))

        predictions = np.delete(predictions, 0, axis=0)
        print('saving txt')
        np.savetxt(
            os.path.join(settings.OUTPUT_FOLDER, fileimg.replace(".jpeg", '.txt')), predictions)
        print('saving jpeg')
        draw = ImageDraw.Draw(fimg)
        i = 1

        while(i < predictions.shape[0]):
            colors = int(255*predictions[i, 4])
            draw.rectangle(predictions[i, 0:4].tolist(), outline=(colors, colors, colors))
            i += 1

        annotations_path = os.path.join(settings.INPUT_FOLDER, fileimg.replace("jpeg", "xml"))
        if draw_annotations and os.path.exists(annotations_path):
            with open(annotations_path) as fd:
                doc = xmltodict.parse(fd.read(), dict_constructor=dict)
                doc1 = deepcopy(doc)
                obj = len(doc1['annotation']['object'])-1

                while(obj != -1):
                    bx1 = int(doc1['annotation']['object'][obj]['bndbox']['xmin'])
                    by1 = int(doc1['annotation']['object'][obj]['bndbox']['ymin'])
                    bx2 = int(doc1['annotation']['object'][obj]['bndbox']['xmax'])
                    by2 = int(doc1['annotation']['object'][obj]['bndbox']['ymax'])
                    draw.rectangle([bx1, by1, bx2, by2], outline=(0, 255, 0))
                    obj -= 1

        fimg.save(os.path.join(settings.OUTPUT_FOLDER, fileimg))
        fimg.close()
        print('done saving')
