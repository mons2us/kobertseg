# """
#     Main training workflow
# """
# from __future__ import division

# import argparse
# import os
# import torch
# import numpy as np
# import random

# from others.logging import init_logger
# from src.train_bertsep import train_sep, test


# def get_args(parser):
#     return parser.parse_args()

# def parse_args(parser):
#     args = get_args(parser)
#     y_dict = load_config(config_path=args.config_path)
#     arg_dict = args.__dict__
#     for key, value in y_dict.items():
#         arg_dict[key] = value
#     return args

# def load_config(config_path='./config.yml'):
#     with open(config_path) as f:
#         configs = yaml.load(f, Loader=yaml.FullLoader)
#     return configs