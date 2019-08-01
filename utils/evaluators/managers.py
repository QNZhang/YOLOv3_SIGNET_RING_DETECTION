# -*- coding: utf-8 -*-
""" utils/evaluators/managers """

from collections import namedtuple

from constants import Dataset
from core.exceptions import DatasetIdInvalid
from .evaluators import COCOAPIEvaluator, SignetRingEvaluator


def get_evaluator_class(dataset_id):
    """ Returns the Evaluator for the dataset """
    if not Dataset.is_valid_option(dataset_id):
        raise DatasetIdInvalid()

    DatasetItem = namedtuple('DatasetItem', ['id', 'evaluator_class'])

    datasets = [
        DatasetItem(Dataset.COCO, COCOAPIEvaluator),
        DatasetItem(Dataset.SIGNET_RING, SignetRingEvaluator),
    ]

    return tuple(filter(lambda x: x.id == dataset_id, datasets))[0].evaluator_class
