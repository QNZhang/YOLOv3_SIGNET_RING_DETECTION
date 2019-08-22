# -*- coding: utf-8 -*-
""" utils/utils """

from __future__ import division
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
import xml.etree.ElementTree as ET

from constants import Dataset
from core.exceptions import DatasetIdInvalid
import settings
from utils.managers.signet_ring_cell_dataset import SignetRingMGR
from .kmeans_iou import kmeans, avg_iou


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
        iou = area / (bbox_area[i] + bbox_area[selec] - area)

        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        class_pred = class_pred[0]
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1,  keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4]*nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def label2yolobox(labels, info_img, maxsize, lrflip):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy = info_img
    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h
    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize
    if lrflip:
        labels[:, 1] = 1 - labels[:, 1]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    h, w, nh, nw, dx, dy = info_img
    y1, x1, y2, x2 = box
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label


def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position

    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape
    img = img[:, :, ::-1]
    assert img is not None

    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw))\
            / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h

    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy+nh, dx:dx+nw, :] = img

    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def random_distort(img, hue, saturation, exposure):
    """
    perform random distortion in the HSV color space.
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        hue (float): random distortion parameter.
        saturation (float): random distortion parameter.
        exposure (float): random distortion parameter.
    Returns:
        img (numpy.ndarray)
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors


def get_tensorboard_log_path(option):
    """ Returns the tensorboard log path for the chosen option """
    if not Dataset.is_valid_option(option):
        raise DatasetIdInvalid()

    log_paths = {
        Dataset.COCO: settings.TENSORBOARD_COCO_LOG_PATH,
        Dataset.SIGNET_RING: settings.TENSORBOARD_SIGNET_LOG_PATH,
    }

    return log_paths[option]


def recalculate_anchor_boxes(option, plot_charts=False, round_centroid_values=True):
    """
    Recalculates and returns the 9 anchor boxes to be used by YOLOv3
    Inspired on: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    Returns numpy.ndarray [9, 2]
    """
    if not Dataset.is_valid_option(option):
        raise DatasetIdInvalid()

    if option == Dataset.COCO:
        # TODO: IMPLEMENT ANCHOR BOXES RECALCULATION FOR COCO
        raise NotImplementedError(
            'Recalculation of anchor boxes for COCO has not been implemented yet.')

    if option == Dataset.SIGNET_RING:
        mgr = SignetRingMGR(settings.SIGNET_BOUNDING_BOXES_PATH)
        boundingboxes_dimentions = tuple([
            (bbox.width, bbox.height) for bbox in mgr.get_annotations(mgr.get_img_ids())])
        kmeans = KMeans(n_clusters=9, random_state=42).fit(boundingboxes_dimentions)
        boundingboxes_dimentions = np.array(boundingboxes_dimentions)
        x_min, y_min = boundingboxes_dimentions[:, 0].min() - 1, \
            boundingboxes_dimentions[:, 1].min() - 1
        x_max, y_max = boundingboxes_dimentions[:, 0].max() + 1, \
            boundingboxes_dimentions[:, 1].max() + 1
        h = 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        centroids = kmeans.cluster_centers_

        if round_centroid_values:
            centroids = np.array(list(map(
                lambda x: [int(round(x[0])), int(round(x[1]))],
                centroids
            )))

        if not plot_charts:
            return centroids

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[14, 9])

        fig.suptitle('K-means clustering on Signet Ring dataset\n'
                     'Centroids are marked with white cross', fontsize=20)

        ax1.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')
        # Plot the centroids as a white X
        ax1.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='r', zorder=10)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        # ax1.set_xticklabels(())
        # ax1.set_yticklabels(())
        ax1.set_xlabel('Bounding box width')
        ax1.set_ylabel('Bounding box height')

        ax2.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        ax2.plot(boundingboxes_dimentions[:, 0], boundingboxes_dimentions[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        ax2.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        # ax2.set_xticklabels(())
        # ax2.set_yticklabels(())
        ax2.set_xlabel('Bounding box width')
        ax2.set_ylabel('Bounding box height')

        # ax3.hist(kmeans.predict(boundingboxes_dimentions), 9, density=True,
        #          facecolor='g', alpha=0.7, edgecolor='black', linewidth=1,
        #          orientation='horizontal')
        # ax3.grid(True)
        # ax3.set_xlabel('Probability')
        # ax3.set_ylabel('Bins')

        pp = pd.DataFrame({'Prediction': kmeans.predict(boundingboxes_dimentions)})
        pp['Prediction'].hist(ax=ax3, grid=True, bins=9)

        plt.tight_layout()
        # Make space for title
        plt.subplots_adjust(top=0.85)
        plt.show()

        return centroids

    raise NotImplementedError


def recalculate_anchor_boxes_kmeans_iou(option, print_results=False, num_centroids=9):
    """
    Recalculates and returns the 9 anchor boxes to be used by YOLOv3
    Inspired on: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    Returns numpy.ndarray [9, 2]

    Inspired on: https://github.com/lars76/kmeans-anchor-boxes/blob/master/example.py
    """
    bndboxes = list()

    if option == Dataset.COCO:
        # TODO: IMPLEMENT ANCHOR BOXES RECALCULATION FOR COCO
        raise NotImplementedError(
            'Recalculation of anchor boxes for COCO has not been implemented yet.')

    if option == Dataset.SIGNET_RING:
        for xml_file in tuple(
                filter(lambda x: x.endswith('.xml'), os.listdir(settings.SIGNET_TRAIN_POS_IMG_PATH))):
            root = ET.parse(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, xml_file)).getroot()
            size = root.find('./size')
            height, width = int(size.find('height').text), int(size.find('width').text)

            for object_ in root.findall('./object'):
                xmin = int(object_.findtext('bndbox/xmin'))  # / width
                ymin = int(object_.findtext('bndbox/ymin'))  # / height
                xmax = int(object_.findtext('bndbox/xmax'))  # / width
                ymax = int(object_.findtext('bndbox/ymax'))  # / height

                bndboxes.append([xmax - xmin, ymax - ymin])

        bndboxes = np.array(bndboxes)
        new_anchors = kmeans(bndboxes, k=num_centroids)

        if print_results:
            print("Accuracy: {:.2f}%".format(avg_iou(bndboxes, new_anchors) * 100))
            print("Boxes:\n {}".format(new_anchors))

            ratios = np.around(new_anchors[:, 0] / new_anchors[:, 1], decimals=2).tolist()
            print("Ratios:\n {}".format(sorted(ratios)))

        return new_anchors

    raise NotImplementedError
