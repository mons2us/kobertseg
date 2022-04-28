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

    # args for tokenizing
    parser.add_argument("-save_path", default='dataset/bbc_news/tokenized_texts')
    parser.add_argument("-raw_path", default='dataset/bbc_news/raw_stories')

    parser.add_argument("-mode", default='', type=str)
    #parser.add_argument("-select_mode", default='greedy', type=str)
    #parser.add_argument("-map_path", default='../../data/')
    parser.add_argument("-dataset_path", default='dataset/')
    parser.add_argument("-data_type", default='bfly', type=str)
    parser.add_argument("-train_ratio", default=0.8, type=float)
    parser.add_argument("-test_only", action='store_true')
    #parser.add_argument("-test_sep_num", default=-1, type=int)

    parser.add_argument("-window_size", default=3, type=int)
    parser.add_argument("-y_ratio", default=0.5, type=float)
    parser.add_argument("-use_stair", action='store_true')
    parser.add_argument("-random_point", action='store_true')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=10, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=10, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    #parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    #parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-log_dir', default='logs/')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=8, type=int)

    args = parser.parse_args()
    log_file = os.path.join(args.log_dir, f'{args.data_type}.log')
    init_logger(log_file)

    # set seed
    set_seed(args.random_seed)

    # !!TODO!! Edit here -> make_data
    eval('data_builder.'+args.mode + '(args)')