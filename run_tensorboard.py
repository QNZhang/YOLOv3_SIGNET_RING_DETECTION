# -*- coding: utf-8 -*-
""" run_tensorboard """

import os
import argparse

from constants import Dataset
from utils.utils import get_tensorboard_log_path


def parse_args():
    """ Parses the arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=int,
        default=Dataset.SIGNET_RING,
        help='Dataset logs to use. Options: {}'.format(Dataset.print_choices())
    )

    return parser.parse_args()


def main():
    """ main """
    args = parse_args()
    path = get_tensorboard_log_path(args.dataset)
    print("Running tensorboard for {} dataset".format(Dataset.print_name(args.dataset)))
    os.system('tensorboard --logdir {}'.format(path))


if __name__ == '__main__':
    main()
