# -*- coding: utf-8 -*-
""" Signet ring cell detection """

import os
import shutil
import yaml
from copy import deepcopy
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import xmltodict
from PIL import Image, ImageDraw
import torch
from torch.autograd import Variable

from constants import BOX_COLOR, Dataset
from models.yolov3 import YOLOv3
from utils.utils import preprocess, postprocess, yolobox2label
from utils.vis_bbox import vis_bbox


###############################################################################
#                                   CONFIGURATION
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
MODEL_CHECKPOINT = 'checkpoints/logs_test_size_0_2_512x512_pos_neg_sample/snapshot9350.ckpt'
CONFIG_FILE = 'config/yolov3_eval_digestpath.cfg'
USE_CUDA = True
CUDA = torch.cuda.is_available() and USE_CUDA
###############################################################################


def initial_validation_cleaning():
    """  """
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(
            "You must create a folder called {} and put the images to be evaluated there."
            .format(INPUT_FOLDER))

    # Cleaning output folder
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)


class MyModel:
    """  """

    def __init__(self):
        self.load_model()

    def load_model(self):
        """  """
        with open(CONFIG_FILE, 'r') as f:
            cfg = yaml.load(f)

        self.imgsize = cfg['TEST']['IMGSIZE']
        self.model = YOLOv3(cfg['MODEL'])
        self.confthre = cfg['TEST']['CONFTHRE']
        self.nmsthre = cfg['TEST']['NMSTHRE']

        if CUDA:
            print("Using cuda")
            self.model = self.model.cuda()

        print("Loading checkpoint {}".format(MODEL_CHECKPOINT))
        state = torch.load(MODEL_CHECKPOINT)
        if 'model_state_dict' in state.keys():
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)

        self.model.eval()

    def get_predictions(self, img_name, plot=False):
        """  """
        img = cv2.imread(os.path.join(INPUT_FOLDER, img_name))
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img, self.imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        if CUDA:
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, Dataset.NUM_CLASSES[Dataset.SIGNET_RING], self.confthre, self.nmsthre)

        if not plot:
            return outputs

        bboxes = list()
        colors = list()

        if outputs[0] is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
                print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
                print('\t+ Conf: %.5f' % cls_conf.item())
                box = yolobox2label([y1, x1, y2, x2], info_img)
                bboxes.append(box)
                colors.append(BOX_COLOR)

        vis_bbox(
            img_raw, bboxes, instance_colors=colors, linewidth=2)
        plt.show()

        return outputs


def main():
    # initial_validation_cleaning()
    pass


if __name__ == '__main__':
    main()
    model = MyModel()
    predictions = model.get_predictions(
        '2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_0.jpeg', True)
    predictions = model.get_predictions(
        '2018_64982_1-3_2019-02-25_21_57_36-lv0-34589-61706-2030-2044_972.8000000000001_0.jpeg',
        True
    )
