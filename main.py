""""
    Main training workflow
"""
from __future__ import division

import argparse
import os
import torch
import numpy as np
import random

from others.logging import init_logger
from src.train_bertsep import train_sep, test

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
                'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(random_seed=227182):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument('--backbone_type', default='bert', type=str, choices=['bert', 'bertsum'])
    parser.add_argument('--classifier_type', default='linear', type=str, choices=['conv', 'linear'])
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--random_seed', default=227182, type=int)

    parser.add_argument('--model_index', default='A01', type=str)
    parser.add_argument('--dataset_path', default='dataset/')
    parser.add_argument('--model_path', default='models/')
    parser.add_argument('--result_path', default='results')
    parser.add_argument('--temp_dir', default='temp/')

    # dataset type
    parser.add_argument('--data_type', default='bfly', type=str)
    parser.add_argument('--window_size', default=3, type=int)
    parser.add_argument('--use_stair', action='store_true')
    parser.add_argument('--random_point', action='store_true')

    parser.add_argument('--batch_size', default=3000, type=int)
    parser.add_argument('--test_batch_size', default=200, type=int)

    parser.add_argument('--max_pos', default=512, type=int)
    parser.add_argument('--use_interval', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--large', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_from_extractive', default='', type=str)

    parser.add_argument('--sep_optim', type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('--lr_bert', default=2e-3, type=float)
    parser.add_argument('--lr_dec', default=2e-3, type=float)
    parser.add_argument('--use_bert_emb', type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('--share_emb', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--finetune_bert', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--dec_dropout', default=0.2, type=float)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dec_hidden_size', default=768, type=int)
    parser.add_argument('--dec_heads', default=8, type=int)
    parser.add_argument('--dec_ff_size', default=2048, type=int)
    parser.add_argument('--enc_hidden_size', default=512, type=int)
    parser.add_argument('--enc_ff_size', default=512, type=int)
    parser.add_argument('--enc_dropout', default=0.2, type=float)
    parser.add_argument('--enc_layers', default=6, type=int)

    # params for sep layers
    parser.add_argument('--add_transformer', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--ext_dropout', default=0.2, type=float)
    parser.add_argument('--ext_layers', default=2, type=int)
    parser.add_argument('--ext_hidden_size', default=768, type=int)
    parser.add_argument('--ext_heads', default=8, type=int)
    parser.add_argument('--ext_ff_size', default=2048, type=int)

    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--generator_shard_size', default=32, type=int)
    parser.add_argument('--alpha',  default=0.6, type=float)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--min_length', default=15, type=int)
    parser.add_argument('--max_length', default=150, type=int)
    parser.add_argument('--max_tgt_len', default=140, type=int)

    parser.add_argument('--param_init', default=0, type=float)
    parser.add_argument('--param_init_glorot', type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--beta1', default= 0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--warmup_steps_bert', default=8000, type=int)
    parser.add_argument('--warmup_steps_dec', default=8000, type=int)
    parser.add_argument('--max_grad_norm', default=0, type=float)

    parser.add_argument('--save_checkpoint_steps', default=500, type=int)
    parser.add_argument('--accum_count', default=1, type=int)
    parser.add_argument('--report_every', default=100, type=int)
    parser.add_argument('--train_steps', default=10000, type=int)
    parser.add_argument('--recall_eval', type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('--valid_steps', default=500, type=int)

    parser.add_argument('--visible_gpus', default='1', type=str)
    parser.add_argument('--train_from', default='', type=str)
    parser.add_argument('--gpu_ranks', default='0', type=str)
    parser.add_argument('--log_dir', default='logs/traineval')

    # Eval
    # !!TODO!! test_from 바꾸기
    parser.add_argument('--test_mode', default='cls', type=str, help="[cls, sep]")
    parser.add_argument('--test_max_mode', default='max_all', type=str, help="[max_all, sep]")
    parser.add_argument('--compare_window', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--test_sep_num', default=2000, type=int)
    parser.add_argument('--test_all', type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('--test_from', default='models/model_w3_fixed_step_50000.pt')
    parser.add_argument('--test_start_from', default=-1, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.visible_gpus.split(',')]
    args.gpu_ranks = [0] if len(args.gpu_ranks) == 1 else args.gpu_ranks
    args.world_size = len(args.gpu_ranks)
    
    # accum_count = n_gpus
    args.accum_count = args.world_size 
    
    print(args.gpu_ranks, args.world_size)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    # set seed
    set_seed(args.random_seed)

    # train
    if args.mode == 'train':
        train_sep(args, device_id)
    elif args.mode == 'test':
        test(args, device_id)