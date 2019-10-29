# -*- coding: utf-8 -*-
""" Signet ring cell detection """

from challenge.classes import MyModel
from challenge.utils import initial_validation_cleaning, process_input_files


def main():
    """
    Processes images from input folder using sliding window technique and saves
    the images along with the predicitons (if configured so) into output folder
    """
    initial_validation_cleaning()
    model = MyModel()
    process_input_files(
        model,
        create_save_img_predictions=True,
        draw_annotations=True
    )


if __name__ == '__main__':
    main()
