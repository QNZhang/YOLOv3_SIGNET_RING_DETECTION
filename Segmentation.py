# -*- coding: utf-8 -*-
""" Signet ring cell detection """

from challenge.classes import MyModel
from challenge.utils import initial_validation_cleaning, process_input_files


def main():
    initial_validation_cleaning()
    model = MyModel()
    process_input_files(model, draw_annotations=False)
    # predictions = model.get_predictions(
    #     img_name='2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_0.jpeg', plot=True)
    # predictions = model.get_predictions(
    #     img_name='2018_64982_1-3_2019-02-25_21_57_36-lv0-34589-61706-2030-2044_972.8000000000001_0.jpeg', plot=True
    # )
    # predictions = model.get_predictions(
    #     img_name='2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_0.jpeg', plot=True)
    # predictions = model.get_predictions(
    #     img_name='2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_204.8.jpeg', plot=True)
    # predictions = model.get_predictions(
    #     img_name='2018_64982_1-3_2019-02-25_21_57_36-lv0-33516-59515-2003-2010_0_409.6.jpeg', plot=True)


if __name__ == '__main__':
    main()
