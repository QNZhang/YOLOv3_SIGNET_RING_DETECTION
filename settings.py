# -*- coding: utf-8 -*-
""" settings """

import os


# Dataset paths
COCO_PATH = 'COCO'


DIGEST_PATH_2019_PATH = '/home/giussepi/Public/link/environments/challenges/digestpath_2019'
SIGNET_TRAIN_NEG_IMG_PATH = os.path.join(
    DIGEST_PATH_2019_PATH, 'digestPath', 'Signet_ring_cell_dataset', 'sig-train-neg')
SIGNET_TRAIN_POS_IMG_PATH = os.path.join(
    DIGEST_PATH_2019_PATH, 'digestPath', 'Signet_ring_cell_dataset', 'sig-train-pos')

SIGNET_BOUNDING_BOXES_PATH = os.path.join(DIGEST_PATH_2019_PATH, 'python_app', 'data', 'signet_bounding_boxes.txt')
# serialized list
SIGNET_TRAIN_PATH = os.path.join(DIGEST_PATH_2019_PATH, 'python_app', 'data', 'signet_train.pickle')
# serialized list
SIGNET_TEST_PATH = os.path.join(DIGEST_PATH_2019_PATH, 'python_app', 'data', 'signet_test.pickle')
