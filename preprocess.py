#encoding=utf-8
import argparse
import time
import os
import re
import torch
import random
import numpy as np

from others.logging import init_logger
import utils.data_builder as data_builder

def set_seed(random_seed=227182):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def do_format_to_lines(args):
    print(time.clock())
    data_builder.format_to_lines(args)
    print(time.clock())

def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())

def do_tokenize(args):
    print(time.clock())
    data_builder.tokenize(args)
    print(time.clock())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)
    parser.add_argument("-random_seed", default=227182, type=int)

    parser.add_argument("-mode", default='generate_sepdata', type=str)
    parser.add_argument("-dataset_path", default='dataset/')
    parser.add_argument("-data_type", default='bfly', type=str)
    parser.add_argument("-train_ratio", default=0.8, type=float)

    parser.add_argument("-window_size", default=3, type=int)
    parser.add_argument("-use_stair", action='store_true')
    parser.add_argument("-random_point", action='store_true')

    parser.add_argument('-min_src_nsents', default=10, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=10, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-log_dir', default='logs/')

    args = parser.parse_args()
    log_file = os.path.join(args.log_dir, f'{args.data_type}.log')
    init_logger(log_file)

    # set seed
    set_seed(args.random_seed)

    # !!TODO!! Edit here -> make_data
    eval('data_builder.' + args.mode + '(args)')