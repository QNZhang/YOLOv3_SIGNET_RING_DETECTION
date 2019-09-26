# -*- coding: utf-8 -*-
""" Signet ring cell detection """

from challenge.classes import MyModel
from challenge.utils import initial_validation_cleaning, process_input_files


def main():
    initial_validation_cleaning()
    model = MyModel()
    process_input_files(
        model,
        create_save_img_predictions=False,
        draw_annotations=False
    )


if __name__ == '__main__':
    main()
