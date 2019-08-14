# -*- coding: utf-8 -*-
""" main """

import os

from constants import Dataset


def main():
    """  """
    # os.system('python demo.py --cfg config/yolov3_default.cfg --image data/mountain.png --detect_thresh 0.5 --weights_path weights/yolov3.weights --dataset={}'.format(Dataset.COCO))

    # os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True --checkpoint_interval=500 --eval_interval=50')
    # os.system('python train.py --tfboard True --checkpoint_interval=500 --eval_interval=50 --checkpoint "checkpoints/test size 0.33/snapshot1500.ckpt"')

    os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --checkpoint "checkpoints/test size 0.33/snapshot500.ckpt"')
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --weights_path weights/yolov3.weights')
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --weights_path weights/darknet53.conv.74 ')

    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.4 --weights weights/yolov3.weights')
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.5 --ckpt checkpoints/snapshot1000.ckpt')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # smallest
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/G1900703-2_2019-04-30 09_53_59-lv0-21164-5136-2060-2007.jpeg" --detect_thresh 0.45 --ckpt "checkpoints/test size 0.2/snapshot500.ckpt"')
    # random image
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.45 --ckpt "checkpoints/test size 0.2/snapshot500.ckpt"')


if __name__ == '__main__':
    main()
