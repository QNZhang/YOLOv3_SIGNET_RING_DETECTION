# -*- coding: utf-8 -*-
""" constants """

from collections import namedtuple
from core.exceptions import DatasetIdInvalid


SIGNET_RING_CLASS_ID = 1
BOX_COLOR = [59, 47, 233]  # blue


DatasetItem = namedtuple('DatasetItem', ['id', 'name'])


class Dataset:
    """ Holds the datasets to work with """
    COCO = 1
    SIGNET_RING = 2

    CHOICES = [
        DatasetItem(COCO, 'COCO'),
        DatasetItem(SIGNET_RING, 'Signet Ring'),
    ]

    NUM_CLASSES = {
        COCO: 80,
        SIGNET_RING: 2,
    }

    @classmethod
    def is_valid_option(cls, dataset_id):
        """ Returns True if the provided id is among the database implementations """
        return dataset_id in [dataset.id for dataset in cls.CHOICES]

    @classmethod
    def print_choices(cls):
        """ Prints the available datasets """
        return ', '.join(tuple('{} - {}'.format(*dataset) for dataset in cls.CHOICES))

    @classmethod
    def print_name(cls, dataset_id):
        """ Prints the name associates with the provided id """
        if not cls.is_valid_option(dataset_id):
            raise DatasetIdInvalid()

        return tuple(filter(lambda x: x.id == dataset_id, cls.CHOICES))[0].name
