# -*- coding: utf-8 -*-
""" utils/kfb """

import os

import json
import kfbReader

import settings


def read_roi_json(roi_json_filepath, kfb_filepath=''):
    """
    * Reads an roi json file and returns its numpy.ndarray plus the roi coordinates
    * If kfb_filepath is not provided, then it's extracted from roi_json_filepath
    Args:
        roi_json_filepath: 'full_path_to_roi_json_file.json'
        kfb_filepath: 'full_path_to_folder_with_kfb_files' (optional)
    Return:
        numpy.ndarray, dictionary
    """
    assert roi_json_filepath.endswith('.json')
    assert os.path.isfile(roi_json_filepath), \
        '{} is not a file or does not exist'.format(roi_json_filepath)
    assert isinstance(kfb_filepath, str), \
        "{} must a be str instance".format(kfb_filepath)

    if not os.path.isdir(kfb_filepath):
        kfb_filepath = os.path.split(roi_json_filepath)[0]

    with open(roi_json_filepath) as _file:
        roi_file = json.load(_file)

    kfb_filepath = os.path.join(kfb_filepath, roi_file['source'])
    assert os.path.isfile(kfb_filepath), \
        "{} image does not exist".format(kfb_filepath)

    read = kfbReader.reader()
    read.setReadScale(settings.KFBREADER_SCALE)
    read.ReadInfo(kfb_filepath, settings.KFBREADER_SCALE, False)

    return read.ReadRoi(
        roi_file['roi']['x'],
        roi_file['roi']['y'],
        roi_file['roi']['w'],
        roi_file['roi']['h'],
        scale=settings.KFBREADER_SCALE
    ), roi_file['roi']
