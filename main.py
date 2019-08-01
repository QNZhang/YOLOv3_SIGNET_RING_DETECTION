# -*- coding: utf-8 -*-
""" main """

import os
import time

from constants import Dataset as dataset_option
import settings
# from utils.evaluators.evaluators import SignetRingEvaluator
# from utils.evaluators.managers import get_evaluator_class
from utils.managers.signet_ring_cell_dataset import SignetRingMGR


def main():
    """  """
    # a = SignetRingMGR(settings.SIGNET_TEST_PATH)
    # a.get_annotations('2018_64982_1-3_2019-02-25 21_57_36-lv0-38368-62991-2040-2016', None)
    # print(get_evaluator_class(1))

    # os.system('python demo.py --image data/mountain.png --detect_thresh 0.5 --weights_path weights/yolov3.weights')

    # os.system('python train.py --weights_path weights/darknet53.conv.74 --tfboard log')
    # os.system('python train.py --weights_path weights/darknet53.conv.74 --n_cpu=12')
    os.system('python train.py --weights_path weights/darknet53.conv.74')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # os.system('python train.py --weights_path weights/darknet53.conv.74 --cfg=config/yolov3_default.cfg --dataset={}'.format(dataset_option.COCO))
    # SIGNET
    # (Pdb) imgs.shape
    # torch.Size([4, 3, 608, 608])
    # (Pdb) targets.shape
    # torch.Size([4, 50, 5])
    #######
    # COCO
    # (Pdb) imgs.shape
    # torch.Size([4, 3, 608, 608])
    # (Pdb) targets.shape
    # torch.Size([4, 50, 5])
    ####
#       File "train.py", line 227, in <module>
#     main()
#   File "train.py", line 186, in main
#     loss = model(imgs, targets)
#   File "/home/giussepi/Public/link/environments/challenges/digestpath_2019/env/lib/python3.6/site-packages/torch/nn/modules/module.py", line 493, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/models/yolov3.py", line 154, in forward
#     x, *loss_dict = module(x, targets)
#   File "/home/giussepi/Public/link/environments/challenges/digestpath_2019/env/lib/python3.6/site-packages/torch/nn/modules/module.py", line 493, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/media/giussepi/Samsung_T5/Desktop info/Public/environments/challenges/digestpath_2019/python_app/third_party/PyTorch_YOLOv3/models/yolo_layer.py", line 176, in forward
#     0].to(torch.int16).numpy()] = 1
# IndexError: index 6 is out of bounds for dimension 4 with size 6

    # a = SignetRingEvaluator(model_type='YOLOV3', img_size=416, confthre=0.8, nmsthre=0.4)
    # dataiterator = iter(a.dataloader)
    # img, _, info_img, id_ = next(dataiterator)  # load a batch


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
