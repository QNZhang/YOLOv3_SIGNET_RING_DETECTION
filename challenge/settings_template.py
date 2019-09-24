# -*- coding: utf-8 -*-
""" challenge settings  """

INPUT_FOLDER = 'input'

OUTPUT_FOLDER = 'output'

MODEL_CHECKPOINT = 'checkpoints/logs_test_size_0_2_512x512_pos_neg_sample/snapshot9350.ckpt'

CONFIG_FILE = 'config/yolov3_eval_digestpath.cfg'

USE_CUDA = True

CUDA = torch.cuda.is_available() and USE_CUDA

cut_size = 512

overlap = 0.4 * cut_size

# holdback = 0.7  # ?????

# smalllim = 0.3  # ?????

boardcache = 2

# save_txt = OUTPUT_FOLDER
