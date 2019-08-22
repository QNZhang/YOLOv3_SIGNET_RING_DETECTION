# -*- coding: utf-8 -*-
""" utils/data """

import os

import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.model_selection import train_test_split

from core.classes import SignetBox
import settings
from .files import get_name_and_extension


def create_bndbox_file():
    """
    - Processes the xml files and creates a dictionary with the format:
      key = image file name
      value = [SignetBox(), SignetBox(), SignetBox(), ...]
    - Uses picle to save the dictionary in the file settings.SIGNET_BOUNDING_BOXES_PATH
    """
    dir_path = os.path.dirname(settings.SIGNET_BOUNDING_BOXES_PATH)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    bndbox_dictionary = defaultdict(list)
    counter = 0

    for xml_file in tuple(
            filter(lambda x: x.endswith('.xml'), os.listdir(settings.SIGNET_TRAIN_POS_IMG_PATH))):
        root = ET.parse(os.path.join(settings.SIGNET_TRAIN_POS_IMG_PATH, xml_file)).getroot()
        img_id, _ = get_name_and_extension(xml_file)
        for bndbox in root.findall('./object'):
            details = dict(
                pose=bndbox.find('pose').text,
                truncated=int(bndbox.find('truncated').text),
                occluded=int(bndbox.find('occluded').text),
                difficult=int(bndbox.find('difficult').text),
            )

            bnd_box = {elem.tag: float(elem.text) for elem in bndbox.find('bndbox').getchildren()}

            counter += 1
            bndbox_dictionary[img_id].append(SignetBox(counter, img_id, details, bnd_box))

    with open(settings.SIGNET_BOUNDING_BOXES_PATH, 'wb') as file_:
        file_.write(pickle.dumps(bndbox_dictionary))


def get_or_create_bndbox_dict(force_create=False):
    """
    - If the file with the bounding boxes does not exits, then it is created.
    - Loads the file using pickle and returns it
    """
    if not os.path.isfile(settings.SIGNET_BOUNDING_BOXES_PATH) or force_create:
        create_bndbox_file()

    bndbox_dictionary = None

    with open(settings.SIGNET_BOUNDING_BOXES_PATH, 'rb') as file_:
        bndbox_dictionary = pickle.load(file_)

    return bndbox_dictionary


def get_or_create_train_test_files(
        test_size=0.33, random_state=42, shuffle=True, force_create=False):
    """  """
    bndbox_dictionary = get_or_create_bndbox_dict(force_create)
    files = [settings.SIGNET_TRAIN_PATH, settings.SIGNET_TEST_PATH]
    lists = []

    if os.path.isfile(files[0]) and os.path.isfile(files[1]) and not force_create:
        for file_ in files:
            with open(file_, 'rb') as file_:
                lists.append(pickle.load(file_))
        return lists[0], lists[1]

    train, test = train_test_split(
        list(bndbox_dictionary.items()), test_size=test_size, random_state=random_state,
        shuffle=shuffle
    )

    train = dict(train)
    test = dict(test)

    for file_path, content in zip(files, [train, test]):
        if not os.path.isfile(file_path) or force_create:
            dir_name = os.path.dirname(file_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            with open(file_path, 'wb') as file_:
                file_.write(pickle.dumps(content))

            lists.append(content)

    return lists[0], lists[1]
