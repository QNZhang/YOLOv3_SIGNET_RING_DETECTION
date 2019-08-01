# -*- coding: utf-8 -*-
""" datasets/managers """

from collections import namedtuple

from constants import Dataset
from core.exceptions import DatasetIdInvalid
from .datasets import COCODataset, SignetRing


def get_dataset_class(dataset_id):
    """ Returns the Dataset manager corresponding to the id provided """
    if not Dataset.is_valid_option(dataset_id):
        raise DatasetIdInvalid()

    DatasetItem = namedtuple('DatasetItem', ['id', 'dataset_class'])

    datasets = [
        DatasetItem(Dataset.COCO, COCODataset),
        DatasetItem(Dataset.SIGNET_RING, SignetRing),
    ]

    return tuple(filter(lambda x: x.id == dataset_id, datasets))[0].dataset_class
