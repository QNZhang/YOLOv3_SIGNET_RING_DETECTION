# -*- coding: utf-8 -*-
""" core/classes """

from collections import namedtuple


ObjDetails = namedtuple('ObjDetails', ['pose', 'truncated', 'occluded', 'difficult'])
BndBox = namedtuple('BndBox', ['xmin', 'ymin', 'xmax', 'ymax', ])


class SignetBox:
    """ Holds data from a signet box  """

    def __init__(self, id_,  image_id, obj_details, bnd_box):
        """ Initializes the object """
        assert isinstance(obj_details, dict)
        assert isinstance(bnd_box, dict)

        self.image_id = image_id
        self.details = ObjDetails(**obj_details)
        self.bndbox = BndBox(**bnd_box)
        self.area = self.width * self.height
        self.id = id_

    def __str__(self):
        """ String representation """
        return str(self.bndbox)

    @property
    def width(self):
        """ Calculates and returns the width of the bounding box """
        return abs(self.bndbox.xmax - self.bndbox.xmin)

    @property
    def height(self):
        """ Calculates and returns the height of the bounding box """
        return abs(self.bndbox.ymax - self.bndbox.ymin)

    def to_dict(self):
        """ Returns the object as a dictionary """
        return dict(
            image_id=self.image_id,
            details=self.details,
            bbox=self.bndbox,
            width=self.width,
            height=self.height,
            area=self.area,
            id=self.id,
        )
