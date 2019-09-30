# -*- coding: utf-8 -*-
""" main """

import os

from constants import Dataset
from utils.utils import recalculate_anchor_boxes, recalculate_anchor_boxes_kmeans_iou
from utils.data import get_or_create_bndbox_dict, get_or_create_train_test_files, \
    create_bndbox_file
from utils.plot_tools import plot_img_plus_bounding_boxes


def main():
    """  """
    # os.system('python demo.py --cfg config/yolov3_default.cfg --image data/mountain.png --detect_thresh 0.5 --weights_path weights/yolov3.weights --dataset={}'.format(Dataset.COCO))

    # os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True --checkpoint_interval=50 --eval_interval=50')
    # os.system('python train.py --tfboard True --checkpoint_interval=50 --eval_interval=50 --checkpoint "checkpoints/logs_test_size_0_2_512x512_pos_neg_sample/snapshot9350.ckpt"')
    os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --checkpoint "checkpoints/logs_test_size_0_2_512x512_pos_neg_sample/snapshot9350.ckpt"')

    #os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --checkpoint "checkpoints/xingru new anchors non-normalizesd IOU kmeans/snapshot750.ckpt"')
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --weights_path weights/yolov3.weights')
    # os.system('python train.py --cfg config/yolov3_eval_digestpath.cfg --eval_interval 1 --weights_path weights/darknet53.conv.74 ')

    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.4 --weights weights/yolov3.weights')
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.5 --ckpt checkpoints/snapshot1000.ckpt')
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos-sliced/2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_0.jpeg" --detect_thresh 0.4 --ckpt checkpoints/snapshot3000.ckpt')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # smallest
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/G1900703-2_2019-04-30 09_53_59-lv0-21164-5136-2060-2007.jpeg" --detect_thresh 0.45 --ckpt "checkpoints/test size 0.2/snapshot500.ckpt"')
    # random image
    # os.system('python demo.py --image "/home/giussepi/Public/link/environments/challenges/digestpath_2019/digestPath/Signet_ring_cell_dataset/sig-train-pos/2018_69188_1-1_2019-03-14 23_40_58-lv0-47611-63515-2072-2046.jpeg" --detect_thresh 0.45 --ckpt "checkpoints/test size 0.2/snapshot500.ckpt"')

    ###############################################################################
    # bndbox_dictionary = get_or_create_bndbox_dict(force_create=True)
    # plot_img_plus_bounding_boxes('G1900703-2_2019-04-30 09_53_59-lv0-6630-14336-2073-2073.jpeg')
    # plot_img_plus_bounding_boxes('2018_68000_1-8_2019-02-26 02_28_01-lv0-83128-38504-2005-2016.jpeg')
    # train, test = get_or_create_train_test_files()
    # plot_img_plus_bounding_boxes('G1900703-2_2019-04-30 09_53_59-lv0-21164-5136-2060-2007.jpeg')
    # plot_img_plus_bounding_boxes('2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_0.jpeg')
    ###############################################################################


if __name__ == '__main__':
    main()
    ###############################################################################
    # train, test = get_or_create_train_test_files(test_size=0.2, force_create=False)
    # bbox = test[list(test.keys())[0]][0]
    # print(bbox.area, bbox.id)
    ###############################################################################
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
    # a = recalculate_anchor_boxes_kmeans_iou(Dataset.SIGNET_RING, print_results=True)
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

    # sliced
    # 9 centroids
    # Accuracy: 82.33%
    # Boxes:
    #     [
    #         [25 50]
    #         [42 44]
    #         [46 29]
    #         [44 66]
    #         [53 54]
    #         [62 42]
    #         [62 72]
    #         [72 57]
    #         [87 88]
    #     ]
    # Ratios:
    #     [0.5, 0.67, 0.86, 0.95, 0.98, 0.99, 1.26, 1.48, 1.59]
    # recalculate_anchor_boxes_kmeans_iou(Dataset.SIGNET_RING, print_results=True, num_centroids=12)
    # Accuracy: 83.95%
    # Boxes:
    #     [
    #         [23, 51],
    #         [36, 30],
    #         [39, 54],
    #         [40, 40],
    #         [48, 47],
    #         [47, 69],
    #         [54, 23],
    #         [56, 55],
    #         [60, 40],
    #         [64, 72],
    #         [72, 55],
    #         [90, 90],
    #     ]
    # Ratios:
    #     [0.45, 0.68, 0.72, 0.89, 1.0, 1.0, 1.02, 1.02, 1.2, 1.31, 1.5, 2.35]

    # recalculate_anchor_boxes_kmeans_iou(Dataset.SIGNET_RING, print_results=True, num_centroids=15)
    # Accuracy: 85.45%
    # Boxes:
    #     [
    #         [22,  49],
    #         [34,  64],
    #         [37,  42],
    #         [44,  29],
    #         [44,  52],
    #         [47,  70],
    #         [50,  43],
    #         [53,  57],
    #         [61,  50],
    #         [66,  36],
    #         [61,  67],
    #         [74,  56],
    #         [67,  87],
    #         [82,  73],
    #         [98, 101],
    #     ]
    # Ratios:
    #     [0.45, 0.53, 0.67, 0.77, 0.85, 0.88, 0.91, 0.93, 0.97, 1.12, 1.16, 1.22, 1.32, 1.52, 1.83]

    # Accuracy: 82.02% - 512x512
    # Boxes:
    #     [
    #         [27, 50],
    #         [42, 40],
    #         [42, 62],
    #         [51, 24],
    #         [52, 51],
    #         [58, 72],
    #         [64, 41],
    #         [69, 58],
    #         [85, 86],
    #     ]
    # Ratios:
    #     [0.54, 0.68, 0.81, 0.99, 1.02, 1.05, 1.19, 1.56, 2.12]

    # Some negative images with detections

    # jpegs = os.listdir(settings.SIGNET_TRAIN_NEG_IMG_PATH)
    # jpegs = list(filter(lambda x: x.endswith('.jpeg'), jpegs))

    # for file_ in jpegs:
    #     path = os.path.join(settings.SIGNET_TRAIN_NEG_IMG_PATH, file_)
    #     os.system('python demo.py --image "{}" --detect_thresh 0.45 --ckpt "checkpoints/logs_test_size_0_2_512x512/snapshot3800.ckpt"'.format(path))

    # /media/giussepi/xingru_dev/Signet_ring_cell_dataset/sig-train-neg-sliced-512-sample/D20190112103_2019-06-10_16_28_08-lv0-29630-6271-2000-2000_1228.8_1228.8.jpeg
    # /media/giussepi/xingru_dev/Signet_ring_cell_dataset/sig-train-neg-sliced-512-sample/D20190441404_2019-06-10_10_16_57-lv0-23312-22369-2000-2000_1228.8_1024.0.jpeg
    # /media/giussepi/xingru_dev/Signet_ring_cell_dataset/sig-train-neg-sliced-512-sample/2019-32176-1-1-1_2019-05-28_11_43_23-lv0-75186-37075-2000-2000_0_1488.jpeg
    # /media/giussepi/xingru_dev/Signet_ring_cell_dataset/sig-train-neg-sliced-512-sample/2019-32177-1-1-1_2019-05-28_11_36_01-lv0-69099-28446-2000-2000_614.4000000000001_1024.0.jpeg
    # /media/giussepi/xingru_dev/Signet_ring_cell_dataset/sig-train-neg-sliced-512-sample/2019-06-11_14_33_26-lv0-34466-8170-2000-2000_819.2_1228.8.jpeg

    ###############################################################################
    # FIRST EXPERIMENTS USING POS AND NEG IMAGES
    ###############################################################################
    # /home/giussepi/Public/environments/challenges/digestpath_2019/env/lib/python3.6/site-packages/numpy/core/fromnumeric.py:86: RuntimeWarning: overflow encountered in reduce
    #   return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    # /media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/utils/utils.py:56: RuntimeWarning: invalid value encountered in subtract
    #   iou = area / (bbox_area[i] + bbox_area[selec] - area)
    # /media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/utils/utils.py:58: RuntimeWarning: invalid value encountered in greater_equal
    #   if (iou >= thresh).any():
    # /media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/utils/utils.py:56: RuntimeWarning: overflow encountered in add
    #   iou = area / (bbox_area[i] + bbox_area[selec] - area)
    # /media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/utils/utils.py:56: RuntimeWarning: invalid value encountered in true_divide
    #   iou = area / (bbox_area[i] + bbox_area[selec] - area)
    # /home/giussepi/Public/environments/challenges/digestpath_2019/env/lib/python3.6/site-packages/numpy/core/fromnumeric.py:86: RuntimeWarning: invalid value encountered in reduce
    #   return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    # /media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/utils/utils.py:55: RuntimeWarning: invalid value encountered in multiply
    #   area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
    # Running per image evaluation...
    # Evaluate annotation type *bbox*
    # DONE (t=13.40s).
    # Accumulating evaluation results...
    # DONE (t=0.83s).
    #  Average Precision  (AP) @[ IoU=0.30      | area=   all | maxDets=400 ] = 0.005
    #  Average Precision  (AP) @[ IoU=0.30      | area= small | maxDets=400 ] = 0.000
    #  Average Precision  (AP) @[ IoU=0.30      | area=medium | maxDets=400 ] = 0.011
    #  Average Precision  (AP) @[ IoU=0.30      | area= large | maxDets=400 ] = 0.001
    #  Average Recall     (AR) @[ IoU=0.30      | area=   all | maxDets=  1 ] = 0.003
    #  Average Recall     (AR) @[ IoU=0.30      | area=   all | maxDets= 10 ] = 0.023
    #  Average Recall     (AR) @[ IoU=0.30      | area=   all | maxDets=400 ] = 0.563
    #  Average Recall     (AR) @[ IoU=0.30      | area= small | maxDets=400 ] = 0.074
    #  Average Recall     (AR) @[ IoU=0.30      | area=medium | maxDets=400 ] = 0.574
    #  Average Recall     (AR) @[ IoU=0.30      | area= large | maxDets=400 ] = 0.976
    # [Iter 50/3500] [lr 0.000000] [Losses: xy 8.029614, wh 11.818830, conf 15137.181641, cls 5.940384, total 5562.636230, imgsize 384]
