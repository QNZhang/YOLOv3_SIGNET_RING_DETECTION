# -*- coding: utf-8 -*-
""" settings """

import os

###############################################################################
#                                   Dataset paths
###############################################################################

COCO_PATH = 'COCO'


DIGEST_PATH_2019_PATH = '/home/giussepi/Public/link/environments/challenges/digestpath_2019'
SIGNET_TRAIN_NEG_IMG_PATH = os.path.join(
    DIGEST_PATH_2019_PATH, 'digestPath', 'Signet_ring_cell_dataset', 'sig-train-neg')
SIGNET_TRAIN_POS_IMG_PATH = os.path.join(
    DIGEST_PATH_2019_PATH, 'digestPath', 'Signet_ring_cell_dataset', 'sig-train-pos-sliced-xingru')

PICKLE_FILES_PATH = os.path.join('data', 'pickle_files')
SIGNET_BOUNDING_BOXES_PATH = os.path.join(PICKLE_FILES_PATH, 'signet_bounding_boxes.pickle')
# serialized list
SIGNET_TRAIN_PATH = os.path.join(PICKLE_FILES_PATH, 'xingru_train.pickle')
# serialized list
SIGNET_TEST_PATH = os.path.join(PICKLE_FILES_PATH, 'xingru_val.pickle')

###############################################################################
#                                   Evaluation
###############################################################################

EVAL_LINEAR_SPACE_LOWER_BOUND = 0.3
EVAL_LINEAR_SPACE_UPPER_BOUND = 0.3  # 0.95
EVAL_LINEAR_SPACE_STEP = 0.05  # spacing between samples
# XXX: After modifying these values, you must do the following updates:
# utils/evaluators/detection_evaluators.py:SignetRingEval.summarize._summarizeDets
#   What is stored on stats variable
# utils/evaluators/evaluators:SignetRingEvaluator.evaluate
#   What it returned taking into account the previous modifications. These two
#   objects will be used when logging values for tensorboardX
# train:main
#   Just update the following lines properly taking into account the previous
#   modifications:
#   ap30, ar30 = evaluator.evaluate(model)
#   tblogger.add_scalar('val/SIGNETAP30', ap30, iter_i)
#   tblogger.add_scalar('val/SIGNETAR30', ar30, iter_i)

###############################################################################
#                                   TENSORBOARD
###############################################################################

TENSORBOARD_SIGNET_LOG_PATH = 'tensorboard_logs/signet_logs'
TENSORBOARD_COCO_LOG_PATH = 'tensorboard_logs/coco_logs'
