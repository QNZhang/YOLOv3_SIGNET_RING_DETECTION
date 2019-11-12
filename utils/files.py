# -*- coding: utf-8 -*-
""" utils/files """

import json
import os
import shutil
from collections import defaultdict

import kfbReader

from challenge.utils import generate_save_xml, settings as challenge_settings


def get_name_and_extension(file_name):
    """
    Returns a tuple with the name and extension
    (name, extension)
    """
    assert isinstance(file_name, str)

    bits = file_name.split('.')

    assert len(bits) >= 2

    return '.'.join(bits[:-1]), bits[-1]


def generate_roi_and_bboxes_files():
    """
    * Reads the json and kfb files from challenge_settings.INPUT_FOLDER
    * Generates the img_roix json and img_roiX xml files and saves them in
      challenge_settings.OUTPUT_FOLDER
      - Each img_roiX json file contains a roi + the image file source name
        {"source": "T2019_78.kfb", "roi": {"x": 22544, "y": 30002, "w": 3584, "h": 3158}}
      - Each img_roiX xml file contains the bboxes (xmin, ymin, xmax, ymax)
    """

    def initial_validation_cleaning():
        """ Verifies the input folder exists and cleans the output folder """
        if not os.path.exists(challenge_settings.INPUT_FOLDER):
            raise FileNotFoundError(
                "You must create a folder called {} and place the kfb and json files there."
                .format(challenge_settings.INPUT_FOLDER))

        # Cleaning output folder
        if os.path.exists(challenge_settings.OUTPUT_FOLDER):
            shutil.rmtree(challenge_settings.OUTPUT_FOLDER)
        os.makedirs(challenge_settings.OUTPUT_FOLDER)

    def annotation_belongs_to_roi(annotation, roi):
        """
        If the annotation lies inside the ROI returns True; otherwise returns False
        """
        ann_x_min = annotation['x']
        ann_y_min = annotation['y']
        ann_x_max = annotation['x'] + annotation['w']
        ann_y_max = annotation['y'] + annotation['h']

        roi_x_min = roi['x']
        roi_y_min = roi['y']
        roi_x_max = roi['x'] + roi['w']
        roi_y_max = roi['y'] + roi['h']

        return (roi_x_min <= ann_x_min <= roi_x_max) and \
            (roi_y_min <= ann_y_min <= roi_y_max) and \
            (roi_x_min <= ann_x_max <= roi_x_max) and \
            (roi_y_min <= ann_y_max <= roi_y_max)

    initial_validation_cleaning()

    scale = 20
    read = kfbReader.reader()
    read.setReadScale(scale)

    for _file in list(filter(
            lambda x: x.endswith('.json'), os.listdir(challenge_settings.INPUT_FOLDER))):
        name, extension = get_name_and_extension(_file)
        kfb_filename = "{}.kfb".format(name)
        read.ReadInfo(
            os.path.join(challenge_settings.INPUT_FOLDER, kfb_filename),
            scale,
            True
        )
        height = read.getHeight()
        width = read.getWidth()

        with open(os.path.join(challenge_settings.INPUT_FOLDER, _file)) as _jfile:
            json_file = json.load(_jfile)
            rois = list(filter(lambda x: x['class'] == 'roi', json_file))
            rois_annotations = defaultdict(list)
            annotations = list(filter(lambda x: x['class'] == 'pos', json_file))

            # splitting annotations into the right roi lists
            for annotation in annotations:
                for index, roi in enumerate(rois):
                    if annotation_belongs_to_roi(annotation, roi):
                        # order of elements xmin, ymin, xmax, ymax, confidence
                        # Note: confidence is set to 1 because it's the ground truth
                        rois_annotations[index].append([
                            annotation['x'],
                            annotation['y'],
                            annotation['x'] + annotation['w'],
                            annotation['y'] + annotation['h'],
                            1
                        ])
                        break

            # creating img_roiX json files and img_roiX xml files
            for index, line in enumerate(rois):
                data = dict(
                    source=kfb_filename,
                    roi={'x': line['x'], 'y': line['y'], 'w': line['w'], 'h': line['h']}
                )
                with open(os.path.join(challenge_settings.OUTPUT_FOLDER, '{}-roi{}.json'.format(name, index+1)), 'w') as roi_json_file:
                    json.dump(data, roi_json_file)

                generate_save_xml(rois_annotations[index], name, width, height, index+1)
