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
    DIGEST_PATH_2019_PATH, 'digestPath', 'Signet_ring_cell_dataset', 'sig-train-pos-sliced')

PICKLE_FILES_PATH = os.path.join('data', 'pickle_files')
SIGNET_BOUNDING_BOXES_PATH = os.path.join(PICKLE_FILES_PATH, 'signet_bounding_boxes.pickle')
# serialized list
SIGNET_TRAIN_PATH = os.path.join(PICKLE_FILES_PATH, 'train.pickle')
# serialized list
SIGNET_TEST_PATH = os.path.join(PICKLE_FILES_PATH, 'test.pickle')

###############################################################################
#                                   Processing folders
###############################################################################

INPUT_FOLDER = 'input'

OUTPUT_FOLDER = 'output'

TEST_INPUT_FOLDER = 'testing'

TEST_OUPUT_FOLDER = 'prediction_result/'

# files generated during Tianchi Testing
TEST_TMP_DATA = 'user_data/jpeg_minipatches/'

###############################################################################
#                                 Minipatches                                  #
###############################################################################

HOLDBACK = 0.7
SMALLLIM = 0.3
CUT_SIZE = 512
OVERLAP_COEFFICIENT = 0.5
# overlap must be an integer to avoid errors with the sliding window algorithm
OVERLAP = int(OVERLAP_COEFFICIENT * CUT_SIZE)
KFBREADER_SCALE = 20

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
