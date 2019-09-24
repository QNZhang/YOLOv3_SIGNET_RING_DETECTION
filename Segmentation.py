# -*- coding: utf-8 -*-
""" Signet ring cell detection """

from challenge.classes import MyModel
from challenge.utils import initial_validation_cleaning


def main():
    # initial_validation_cleaning()
    pass


if __name__ == '__main__':
    main()
    model = MyModel()
    predictions = model.get_predictions(
        '2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_0.jpeg', True)
    predictions = model.get_predictions(
        '2018_64982_1-3_2019-02-25_21_57_36-lv0-34589-61706-2030-2044_972.8000000000001_0.jpeg',
        True
    )
