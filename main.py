# -*- coding: utf-8 -*-
""" main """

import os

from constants import Dataset
from utils.utils import recalculate_anchor_boxes, recalculate_anchor_boxes_kmeans_iou


def main():
    """  """
    # os.system('python demo.py --cfg config/yolov3_default.cfg --image data/mountain.png --detect_thresh 0.5 --weights_path weights/yolov3.weights --dataset={}'.format(Dataset.COCO))

    os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True --checkpoint_interval=500 --eval_interval=50')
    # os.system('python train.py --tfboard True --checkpoint_interval=500 --eval_interval=50 --checkpoint "checkpoints/test size 0.33/snapshot1500.ckpt"')

    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --checkpoint "checkpoints/test size 0.33/snapshot500.ckpt"')
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
    # main()
    # a = recalculate_anchor_boxes(Dataset.SIGNET_RING, plot_charts=True, round_centroid_values=True)
    # print(a)
    # [[ 54  47] S
    #  [113 114] L
    #  [ 88  70] L
    #  [ 71  47] M
    #  [ 59  80] M
    #  [ 65  62] M
    #  [ 40  41] S
    #  [ 46  60] S
    #  [ 81  96]] L
    a = recalculate_anchor_boxes_kmeans_iou(Dataset.SIGNET_RING, print_results=True)
    # Accuracy: 85.04%
    # Boxes:
    # [[0.02179297 0.02423954]
    # [0.02649657 0.02078179]
    # [0.0371471  0.0313277 ]
    # [0.03379722 0.02372714]
    # [0.02287419 0.03188406]
    # [0.02798601 0.02768166]
    # [0.04426288 0.04516129]
    # [0.03037727 0.03738318]
    # [0.01895965 0.01858191]]
    # Ratios:
    #  [0.72, 0.81, 0.9, 0.98, 1.01, 1.02, 1.19, 1.27, 1.42]
    # In [4]: a*2000
    # Out[4]:
    # array([[43.58593363, 48.47908745],
    #        [52.99313052, 41.56358238],
    #        [74.29420505, 62.65539533],
    #        [67.5944334 , 47.45427583],
    #        [45.74838389, 63.76811594],
    #        [55.97201399, 55.3633218 ],
    #        [88.52576422, 90.32258065],
    #        [60.75453209, 74.76635514],
    #        [37.91929995, 37.16381418]])

    # Accuracy: 85.00%
    # Boxes:
    #  [[90 93]
    #  [44 64]
    #  [76 66]
    #  [59 73]
    #  [52 54]
    #  [39 37]
    #  [56 42]
    #  [43 47]
    #  [65 53]]
    # Ratios:
    #  [0.69, 0.81, 0.91, 0.96, 0.97, 1.05, 1.15, 1.23, 1.33]

    # Accuracy: 85.14%
    # Boxes:
    #  [[68 46] M
    #  [44 52] S
    #  [49 68] M
    #  [58 56] M
    #  [91 92] L
    #  [65 76] L
    #  [40 39] S
    #  [53 44] S
    #  [75 62]] L
    # Ratios:
    #  [0.72, 0.85, 0.86, 0.99, 1.03, 1.04, 1.2, 1.21, 1.48]
