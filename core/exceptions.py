# -*- coding: utf-8 -*-
""" core/exceptions """


class DatasetIdInvalid(Exception):
    """
    Exception to be raised when an id provided does not belong to any of
    the datasets implemented
    """

    def __init__(self, message=''):
        """  """
        if not message:
            from constants import Dataset
            message = 'The id provided is not a the valid option: {}'.format(
                Dataset.print_choices())
        super().__init__(message)
