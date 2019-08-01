# -*- coding: utf-8 -*-
""" utils/managers/signet_ring_cell_dataset """

import os
import pickle

import settings


class SignetRingMGR:
    """ Handles read operations over signet data """

    def __init__(self, pickle_file=settings.SIGNET_TRAIN_PATH):
        """
        Initializes the object loading the data from the pickle file provided
        """
        if not os.path.isfile(pickle_file):
            raise FileNotFoundError('{} not found'.format(pickle_file))

        with open(pickle_file, 'rb') as file_:
            self.data = pickle.load(file_)

    def get_img_ids(self):
        """ Returns a tuple containing the keys/image names """
        return tuple(self.data.keys())

    def get_annotations(self, index, filters=None, to_dict=False):
        """
        # TODO: Review if the filters really works
        # TODO: all possible values for the filters
        - Gets the SignetBoxes belonging to the index provided
        - Returns a list containing the SignetBoxes filtered by the filters provided
        - If to_dict = True, returns a list with a dictionary representation of the SigntBoxes
        Args:
            index: str or list or tuple,
            filters:  None or dict e.g. dict(pose='', truncated=None, occluded=None,  difficult=None)

        Returns:
          [SignetBox, SignetBox, SignetBox, ...]

          Or if to_dict=True
          [dict, dict, dict, ...]
        """
        assert isinstance(index, (str, list, tuple))
        index_list = [index] if isinstance(index, str) else index

        if filters is None:
            filters = dict(pose='', truncated=None, occluded=None,  difficult=None)

        options = (1, 0, None)

        assert isinstance(filters['pose'], str)
        assert filters['truncated'] in options
        assert filters['occluded'] in options
        assert filters['difficult'] in options

        filters = tuple(filter(lambda x: x[1] not in (None, ''), filters.items()))
        filtered_bndboxes = []

        for index_ in index_list:
            for signetbox in self.data[index_]:
                for key, value in filters:
                    if getattr(signetbox.details, key) != value:
                        break
                else:
                    if to_dict:
                        signetbox = signetbox.to_dict()
                    filtered_bndboxes.append(signetbox)

        return filtered_bndboxes
