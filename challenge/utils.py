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

from utils.utils import nms
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


def generate_save_xml(predictions, fileimg, img_width, img_height, roi_counter):
    """
    Saves the predicitons into an xml file similar to Signet Ring training XML files annotations
    """
    def get_object_dict(xmin, ymin, xmax, ymax, confidence):
        """  """
        return {
            'object': {
                'name': 'cervical_cancer',
                'pose': 'Right',
                'truncated': 1,
                'occluded': 0,
                'confidence': confidence,
                'bndbox': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                },
                'difficult': 0
            }
        }

    with open(
            os.path.join(settings.OUTPUT_FOLDER, "{}-roi{}.xml".format(fileimg, roi_counter)),
            'a'
    ) as file_:
        file_.write('<annotation>\n')
        file_.write(' <folder>annotations</folder>\n')
        file_.write(' <filename>{}.kfb</filename>\n'.format(fileimg))
        file_.write(' <source>\n')
        file_.write('  <database>Cervical Cancer Database</database>\n')
        file_.write('  <annotation>Cancer Dataset</annotation>\n')
        file_.write('  <image>tianchi</image>\n')
        file_.write(' </source>\n')
        file_.write(' <size>\n')
        file_.write('  <width>{}</width>\n'.format(img_width))
        file_.write('  <height>{}</height>\n'.format(img_height))
        file_.write('  <depth>3</depth>\n')
        file_.write(' </size>\n')
        file_.write(' <segmented>0</segmented>\n')

        # creating bounding boxes
        for prediction in predictions:
            file_.write(' ' + xmltodict.unparse(
                get_object_dict(*prediction), pretty=True, full_document=False,
                newl="\n", indent=" "
            ) + '\n')

        file_.write('</annotation>')


def evaluation(x, y, cut_size, w, h, fimg, model):
    """
    Creates the mini-patch and gets its bounding boxes predictions, then transform them
    into the right coordinates in the whole image and return them in a numpy array.

    Returns:

    [[x1, y1, x2, y2, score], ...]

    """
    fimg = cv2.imread(fimg.filename)
    image = fimg[y:y+cut_size, x:x+cut_size]
    # cv2.imshow("cropped", image)

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


def process_input_files(model, create_save_img_predictions=False, draw_annotations=False):
    """
    * Iterates over the images in settings.INPUT_FOLDER
    * Gets the bouding boxes predictions using the sliding window technique
    * Applies non maximum suppression
    * Saves the predictions on settings.OUTPUT_FOLDER and optionally images with
      the predictions and ground truth bounding boxes
    """
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

        # applying non maximum suppression
        selected_ids = nms(predictions[:, :4], model.nmsthre, predictions[:, 4])
        predictions = predictions[selected_ids]

        print('saving xml')
        generate_save_xml(predictions, fileimg, fimg.width, fimg.height)

        if create_save_img_predictions:
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
