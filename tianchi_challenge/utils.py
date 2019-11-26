# -*- coding: utf-8 -*-
""" tianchi_challenge/utils """

import os
import re
import shutil

import cv2
import json
import kfbReader
import numpy as np
from PIL import Image

import settings
from utils.files import get_name_and_extension
from utils.utils import nms


def initial_validation_cleaning():
    """ Verifies the input folder exists and cleans the output folder """
    if not os.path.exists(settings.TEST_INPUT_FOLDER):
        raise FileNotFoundError(
            "You must create a folder called {} and put the images to be evaluated there."
            .format(settings.TEST_INPUT_FOLDER))

    # Cleaning output folder
    if os.path.exists(settings.TEST_OUPUT_FOLDER):
        shutil.rmtree(settings.TEST_OUPUT_FOLDER)
    os.makedirs(settings.TEST_OUPUT_FOLDER)


def evaluation(x, y, cut_size, w, h, fimg, model):
    """
    Gets image bounding boxes predictions, then transform them
    into the right coordinates in the whole image and return them in a numpy array.

    Returns:

    [[x1, y1, x2, y2, score], ...]

    """
    if settings.USE_ROIS:
        # image = fimg[y:y+cut_size, x:x+cut_size]
        image = fimg.ReadRoi(x, y, cut_size, cut_size, scale=settings.KFBREADER_SCALE)
    else:
        image = cv2.imread(fimg.filename)

    results = model.get_predictions(image=image, plot=False)

    if len(results) == 0:
        return None

    c = results.cpu().numpy()

    if(x != 0 and y != 0 and x+cut_size != w and y+cut_size != h):
        i = 0
        while i < c.shape[0]:
            if(c[i, 0] < settings.BOARDCACHE or c[i, 1] < settings.BOARDCACHE or
               c[i, 2] < settings.BOARDCACHE or c[i, 3] < settings.BOARDCACHE):
                c = np.delete(c, i, axis=0)
                i -= 1
            i += 1
    i = 0

    while i < c.shape[0]:
        c[i, 0] += x
        c[i, 1] += y
        c[i, 2] += x
        c[i, 3] += y
        i += 1

    return c


def generate_save_json(predictions, fileimg):
    """  """
    def get_object_dict(xmin, ymin, xmax, ymax, confidence):
        """  """
        return {
            'x': xmin,
            'y': ymin,
            'w': xmax - xmin,
            'h': ymax - ymin,
            'p': confidence
        }

    pred_list = []
    for prediction in predictions:
        pred_list.append(get_object_dict(*prediction))

    with open(os.path.join(settings.TEST_OUPUT_FOLDER, fileimg), 'w') as json_file:
        json.dump(pred_list, json_file)


def process_input_files_jpeg(model):
    """
    * Iterates over the images in settings.INPUT_FOLDER
    * Gets the bouding boxes predictions using the sliding window technique
    * Applies non maximum suppression
    * Saves the predictions on settings.OUTPUT_FOLDER and optionally images with
      the predictions and ground truth bounding boxes
    """
    coord_pattern = re.compile(r'[\w_]+_(?P<x>\d+)_(?P<y>\d+).jpeg')

    for kfbfile in os.listdir(settings.TEST_INPUT_FOLDER):
        name, _ = get_name_and_extension(kfbfile)
        predictions = [[0, 0, 0, 0, 0]]

        for fileimg in tuple(filter(lambda x: x.startswith(name), os.listdir(settings.TEST_TMP_DATA))):
            fimg = Image.open(os.path.join(settings.TEST_TMP_DATA, fileimg))
            # need to get X and Y

            x = int(coord_pattern.match(fileimg, re.IGNORECASE).group('x'))
            y = int(coord_pattern.match(fileimg, re.IGNORECASE).group('y'))
            w, h = fimg.size

            eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, fimg, model)
            fimg.close()

            if eval_results is not None:
                predictions = np.vstack((predictions, eval_results))

        predictions = np.delete(predictions, 0, axis=0)

        # applying non maximum suppression
        selected_ids = nms(predictions[:, :4], model.nmsthre, predictions[:, 4])
        predictions = predictions[selected_ids]
        generate_save_json(predictions, '{}.json'.format(name))


def process_input_files_roi_json(model):
    """
    * Iterates over the images in settings.INPUT_FOLDER
    * Gets the bouding boxes predictions using the sliding window technique
    * Applies non maximum suppression
    * Saves the predictions on settings.OUTPUT_FOLDER
    """
    read = kfbReader.reader()
    read.setReadScale(settings.KFBREADER_SCALE)

    for kfbfile in os.listdir(settings.TEST_INPUT_FOLDER):
        print(kfbfile)
        predictions = [[0, 0, 0, 0, 0]]
        read.ReadInfo(os.path.join(settings.TEST_INPUT_FOLDER, kfbfile), settings.KFBREADER_SCALE, False)
        w, h = read.getWidth(), read.getHeight()
        y = 0

        while(y <= (h-settings.CUT_SIZE)):
            x = 0

            while(x <= (w-settings.CUT_SIZE)):
                eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, read, model)
                if eval_results is not None:
                    predictions = np.vstack((predictions, eval_results))
                x = x+settings.OVERLAP
                # print(x)

            x = w - settings.CUT_SIZE
            eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, read, model)
            if eval_results is not None:
                predictions = np.vstack((predictions, eval_results))
            y = y+settings.OVERLAP
            # print(y)

        x = 0
        y = h - settings.CUT_SIZE

        while(x <= (w-settings.CUT_SIZE)):
            eval_results = evaluation(x, y, settings.CUT_SIZE, w, h, read, model)
            if eval_results is not None:
                predictions = np.vstack((predictions, eval_results))
            x = x+settings.OVERLAP
            # print(x)

        eval_results = evaluation(
            w-settings.CUT_SIZE, h-settings.CUT_SIZE, settings.CUT_SIZE, w, h, read, model)
        if eval_results is not None:
            predictions = np.vstack((predictions, eval_results))

        predictions = np.delete(predictions, 0, axis=0)

        # applying non maximum suppression
        selected_ids = nms(predictions[:, :4], model.nmsthre, predictions[:, 4])
        predictions = predictions[selected_ids]

        print('saving JSON')
        name, _ = get_name_and_extension(kfbfile)
        generate_save_json(predictions, '{}.json'.format(name))
        print('done saving')


def process_input_files(model):
    """ Process the inputs files properly based on settings.USE_ROIS option """
    if settings.USE_ROIS:
        process_input_files_roi_json(model)
    else:
        process_input_files_jpeg(model)
