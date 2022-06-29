#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch

from utils import distributed
from src.trainer import build_trainer
from src.evaluator import build_evaluator, build_sep_evaluator
#from models.trainer_ext import build_trainer
from utils.data_loader import DataLoader, load_dataset
from backbone import model_builder
from backbone.model_builder import BertSeparator
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def train_multi_sep(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        print(f'gpu number: {i}')
        device_id = i
        procs.append(mp.Process(target=run, args=(args, device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()
        

def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])
    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in Distributed initialization")
        train_single_sep(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can prFobably
                be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


# ------------------------
#      Build Trainer
# ------------------------
# add multi if use multi-gpu
def train_sep(args, device_id):
    train_multi_sep(args)
    #train_single_sep(args, device_id)

def train_single_sep(args, device_id):
    os.makedirs(f'models/index_{args.model_index}', exist_ok=True)

    log_dir = f'models/index_{args.model_index}'
    logger_name = f'{args.mode}_{args.model_index}_'
    logger_time = '_'.join(time.asctime().split()[1:4])
    logger_pth = os.path.join(log_dir, logger_name + logger_time + '.log')
    init_logger(logger_pth)

    # device
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    # Load checkpoint to train from
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    # Return data loader to train with
    # !!TODO!! Remove is_test, it's of no use for separation
    def train_iter_fct():
        return DataLoader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                        shuffle=True, is_test=False)

    def valid_iter_fct():
        return DataLoader(args, load_dataset(args, 'valid', shuffle=False), args.batch_size, device,
                        shuffle=False, is_test=False)

    model = BertSeparator(args, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)

    #logger.info(model)
    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps, valid_iter_fct, args.valid_steps)


def test(args, device_id):
    def _get_hyperparams(args, checkpoint):
        opts = checkpoint['opt']
        args.add_transformer = opts.add_transformer
        args.classifier_type = opts.classifier_type
        #args.classifier_type = 'conv'
        args.window_size = opts.window_size
        return args

    logger_pth = os.path.join(args.log_dir, '_'.join(time.asctime().split()) + '.log')
    init_logger(logger_pth)
    
    assert args.test_from
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    
    logger.info('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)

    args = _get_hyperparams(args, checkpoint)

    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    def test_iter_fct():
        return DataLoader(args, load_dataset(args, 'test', shuffle=True), args.batch_size, device,
                        shuffle=True, is_test=False)
    
    model = BertSeparator(args, device_id, checkpoint)

    # Evaluation mode
    if args.test_mode == 'cls':
        evaluator = build_evaluator(args, device_id, model)
        evaluator.cls_eval(test_iter_fct)
    elif args.test_mode == 'sep':
        evaluator = build_sep_evaluator(args, device_id, model)
        evaluator.sep_eval()