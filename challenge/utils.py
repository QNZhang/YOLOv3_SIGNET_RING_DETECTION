# -*- coding: utf-8 -*-
""" challenge utils  """

import os
import shutil

import numpy as np
import torch

from . import settings


def initial_validation_cleaning():
    """  """
    if not os.path.exists(settings.INPUT_FOLDER):
        raise FileNotFoundError(
            "You must create a folder called {} and put the images to be evaluated there."
            .format(settings.INPUT_FOLDER))

    # Cleaning output folder
    if os.path.exists(settings.OUTPUT_FOLDER):
        shutil.rmtree(settings.OUTPUT_FOLDER)
    os.makedirs(settings.OUTPUT_FOLDER)


def use_cuda():
    return torch.cuda.is_available() and settings.USE_CUDA


# def evaluation(x, y, cut_size, w, h, fimg):
#     image1 = fimg.crop((x, y, x+cut_size, y+cut_size))

#     pil_image = image1.convert("RGB")

#     image = np.array(pil_image)[:, :, [2, 1, 0]]

#     #######
#     results = coco_demo.compute_prediction(image)
#     c = results.bbox.cpu().numpy()
#     print(c)
#     #######
# #     c = coco_demo.select_top_predictions(c)
#     if(x != 0 and y != 0 and x+cut_size != w and y+cut_size != h):
#         i = 0
#         while(i < c.shape[0]):
#             if(c[i, 0] < boardcache or c[i, 1] < boardcache or c[i, 2] < boardcache or c[i, 3] < boardcache):
#                 c = np.delete(c, i, axis=0)
#                 i -= 1
#             i += 1
#     i = 0

#     c = np.hstack((c, np.zeros((c.shape[0], 1))))
#     while(i < c.shape[0]):
#         c[i, 0] += x
#         c[i, 2] += x
#         c[i, 1] += y
#         c[i, 3] += y
#         c[i, 4] = results.extra_fields['scores'].cpu().numpy()[i].item()
#         i += 1
#     return c
