# -*- coding: utf-8 -*-
""" main """

import os

import settings
from constants import Dataset
from utils.classes.cutpatch import MiniPatch, TestMiniPatch
from utils.data import get_or_create_bndbox_dict, get_or_create_train_test_files, \
    create_bndbox_file
from tianchi_challenge.classes import MyModel
from tianchi_challenge.utils import initial_validation_cleaning, process_input_files
from utils.files import generate_roi_and_bboxes_files
from utils.plot_tools import plot_img_plus_bounding_boxes
from utils.utils import recalculate_anchor_boxes, recalculate_anchor_boxes_kmeans_iou, dhash, hamming


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


if __name__ == '__main__':
    main()
    ###############################################################################
    # train, test = get_or_create_train_test_files(test_size=0.2, force_create=True)
    # bbox = test[list(test.keys())[0]][0]
    # print(bbox.area, bbox.id)
    ###############################################################################

    # generate_roi_and_bboxes_files()
    # MiniPatch()()

    # recalculate_anchor_boxes_kmeans_iou(Dataset.SIGNET_RING, print_results=True, num_centroids=9)

    ###########################################################################
    #                               Train model                               #
    ###########################################################################

    # os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True --checkpoint_interval=50 --eval_interval=50')
    # os.system('python train.py --tfboard True --checkpoint_interval=50 --eval_interval=50 --checkpoint "checkpoints/tianchi/512x512_roi_confthre_0_005/snapshot3500.ckpt"')

    # os.system('python demo.py --image "/home/giussepi/Downloads/tianchi/positives/T2019_53-roi1_6620_22337.json" --detect_thresh 0.4 --weights weights/yolov3.weights')
    # os.system(
    #     'python demo.py --image "{}" --detect_thresh 0.4 --weights weights/yolov3.weights'
    #     .format(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, 'T2019_4-roi2_16240_25404.jpeg'))
    # )

    ###########################################################################
    #          Creating jpeg of minipatches for training and testing          #
    ###########################################################################
    # from utils.plot_tools import create_X_cervical_images_plus_bounding_boxes
    # create_X_cervical_images_plus_bounding_boxes((22187, 22188), draw_bbox=False)

    ###########################################################################
    #                             TIANCHI TESTING                             #
    ###########################################################################

    initial_validation_cleaning()
    process_input_files(MyModel())

    ###########################################################################
    #                              traing kkmeans                          #
    ###########################################################################
    # a = dhash(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, 'T2019_999-roi2_33232_15350.jpeg'))
    # b = dhash(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, 'T2019_10-roi1_10994_4665.jpeg'))
    # hamming(a, b)

    # from scipy.spatial.distance import cdist
    # # from utils.kmeans import kmeans_hamming
    # import random

    # points2 = [[random.randint(1, 100000)] for _ in range(10)]
    # # points2.extend([random.randint(1, 100000)*10+10 for _ in range(10)])
    # # print(kmeans_hamming(points2, points2[:2], 100))

    # from sklearn.cluster import KMeans
    # import numpy as np

    # kmeans = KMeans(n_clusters=2, random_state=42).fit(points2)
    # TestMiniPatch()()
