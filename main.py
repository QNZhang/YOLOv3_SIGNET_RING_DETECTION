# -*- coding: utf-8 -*-
""" main """

import os


def main():
    """  """
    # os.system('python demo.py --image data/mountain.png --detect_thresh 0.5 --weights_path weights/yolov3.weights')

    # os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard log')
    # os.system('python train.py --weights_path weights/darknet53.conv.74 --n_cpu=12')
    os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard True')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # os.system('python train.py --weights_path weights/darknet53.conv.74 --cfg=config/yolov3_default.cfg --dataset={}'.format(dataset_option.COCO))


if __name__ == '__main__':
    # start = time.time()
    main()
    # end = time.time()
    # print(end - start)
    # 30.752382516860962 cuda cpu 1
    # 30.752382516860962 cuda cpu 12
    # MAXITER = 24(hrs)*3600(segs)/12.729515(segs) = 6787.375638427702
    # run half million epoaches:  500000*12.729515/(3600*24) = 73.66617476851852 days

    # 10 -> 64.96263599395752 cuda cpu 1
    # 10 -> 98.67466306686401 cuda cpu 12
    # 1 -> 228.9973545074463 no cuda cpu 1
    # 1 - > 12.729515075683594
    # 10 - > 73.60321974754333 cuda cpu 0
    # 10 -> 91.3451087474823 - 71.3  cuda cpu 0 benchmark = True
    # 10 -> 79.45707488059998 cuda cpu 12
    # 10 -> 66.49381160736084 - 78.000 cuda cpu 12 benchmark = True
