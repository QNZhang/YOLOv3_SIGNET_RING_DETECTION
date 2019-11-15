# -*- coding: utf-8 -*-
""" utils/kfb """

import os

import json
import kfbReader

import settings


def read_roi_json(roi_json_filepath, kfb_filepath=''):
    """
    * Reads an roi json from a kfb file and returns its numpy.ndarray
    * If kfb_filepath is not provided, then it's extracted from roi_json_filepath
    """
    assert(os.path.isfile(roi_json_filepath))
    assert(isinstance(kfb_filepath, str))

    if not os.path.isdir(kfb_filepath):
        kfb_filepath = os.path.split(roi_json_filepath)[0]

    with open(roi_json_filepath) as _file:
        roi_file = json.load(_file)

    kfb_filepath = os.path.join(kfb_filepath, roi_file['source'])
    assert(os.path.isfile(kfb_filepath))

    read = kfbReader.reader()
    read.setReadScale(settings.KFBREADER_SCALE)
    read.ReadInfo(kfb_filepath, settings.KFBREADER_SCALE, True)

    return read.ReadRoi(
        roi_file['roi']['x'],
        roi_file['roi']['y'],
        roi_file['roi']['w'],
        roi_file['roi']['h'],
        scale=settings.KFBREADER_SCALE
    )
