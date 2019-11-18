# -*- coding: utf-8 -*-
"""datasets/datasets"""

import os

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset

import constants
import settings
from utils.kfb import read_roi_json
from utils.managers.signet_ring_cell_dataset import SignetRingMGR
from utils.utils import label2yolobox, preprocess, random_distort


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self, model_type, data_dir=settings.COCO_PATH,
                 json_file='instances_train2017.json',
                 name='train2017', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset

        source: https://github.com/DeNA/PyTorch_YOLOv3/blob/master/dataset/cocodataset.py
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.model_type = model_type
        self.coco = COCO(os.path.join(self.data_dir, 'annotations/', self.json_file))
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True

        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)
        assert img is not None

        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)

        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_


class SignetRing(Dataset):
    """ SignerRing dataset class """

    def __init__(self, model_type,
                 train_path=settings.SIGNET_TRAIN_PATH,
                 img_train_dir=settings.SIGNET_TRAIN_POS_IMG_PATH,
                 img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        SignetRing dataset initialization.
        Args:
            model_type (str): model name specified in config file
            train_path (str): path to pickle file with training data
            img_train_dir (str): path to folder containing training images
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset

        Inspired on: https://github.com/DeNA/PyTorch_YOLOv3/blob/master/dataset/cocodataset.py
        """
        self.model_type = model_type
        self.img_train_dir = img_train_dir
        self.signet = SignetRingMGR(train_path)
        self.ids = self.signet.get_img_ids()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        # TODO: Review if this max_labels is ok with the number of bndboxes per picture
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        # anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        # annotations = self.coco.loadAnns(anno_ids)
        annotations = self.signet.get_annotations(
            id_,
            filters=dict(pose='', truncated=None, occluded=None,  difficult=None)
        )

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip is True:
            lrflip = True

        img_file = os.path.join(self.img_train_dir, '{}.json'.format(id_))
        img, _ = read_roi_json(img_file)

        assert img is not None

        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)

        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        for anno in annotations:
            # if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
            if anno.width > self.min_size and anno.height > self.min_size:
                labels.append([constants.SIGNET_RING_CLASS_ID])
                labels[-1].extend([
                    anno.bndbox.xmin,
                    anno.bndbox.ymin,
                    anno.width,
                    anno.height,
                ])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_
