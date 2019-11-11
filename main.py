# -*- coding: utf-8 -*-
""" main """

import os

import cv2 as cv
import kfbReader

from constants import Dataset
from utils.utils import recalculate_anchor_boxes, recalculate_anchor_boxes_kmeans_iou
from utils.data import get_or_create_bndbox_dict, get_or_create_train_test_files, \
    create_bndbox_file
from utils.plot_tools import plot_img_plus_bounding_boxes


def main():
    """  """
    # os.system('python demo.py --cfg config/yolov3_default.cfg --image data/mountain.png --detect_thresh 0.5 --weights_path weights/yolov3.weights --dataset={}'.format(Dataset.COCO))

    # os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True --checkpoint_interval=50 --eval_interval=50')
    # os.system('python train.py --tfboard True --checkpoint_interval=50 --eval_interval=50 --checkpoint "checkpoints/confthre_0_dot_8/snapshot12500.ckpt"')

    # hereee
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --checkpoint "checkpoints/confthre_0_dot_8/snapshot17350.ckpt"')
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --weights_path weights/yolov3.weights')
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --weights_path weights/darknet53.conv.74 ')

    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.4 --weights weights/yolov3.weights')
    ###############################################################################
    # bndbox_dictionary = get_or_create_bndbox_dict(force_create=True)
    # plot_img_plus_bounding_boxes('G1900703-2_2019-04-30 09_53_59-lv0-6630-14336-2073-2073.jpeg')
    # train, test = get_or_create_train_test_files()
    # plot_img_plus_bounding_boxes('G1900703-2_2019-04-30 09_53_59-lv0-21164-5136-2060-2007.jpeg')
    ###############################################################################

    # Testing kfbreader #######################################################
    path = "/home/giussepi/Downloads/tianchi/neg_0/T2019_121.kfb"
    scale = 20

    read = kfbReader.reader()
    read.ReadInfo(path, scale, True)
    roi = read.ReadRoi(10240, 10240, 512, 512, scale)
    cv.imshow('roi', roi)
    cv.waitKey(0)

    # height = read.getHeight()
    # width = read.getWidth()
    # scale = read.getReadScale()
    # print('height: ', height)
    # print('width: ', width)
    # print('scale: ', scale)

    # # read.setReadScale(scale=20)

    # roi_0 = read.ReadRoi(5120, 5120, 256, 256, scale=5)
    # roi_1 = read.ReadRoi(10240, 10240, 512, 512, scale=10)
    # roi_2 = read.ReadRoi(20480, 20480, 1024, 1024, scale=20)

    # cv.imshow('roi', roi_0)
    # cv.waitKey(0)
    # cv.imshow('roi', roi_1)
    # cv.waitKey(0)
    # cv.imshow('roi', roi_2)
    # cv.waitKey(0)


if __name__ == '__main__':
    main()
    ###############################################################################
    # train, test = get_or_create_train_test_files(test_size=0.2, force_create=False)
    # bbox = test[list(test.keys())[0]][0]
    # print(bbox.area, bbox.id)
    ###############################################################################
