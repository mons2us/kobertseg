import argparse
import subprocess
import os 
import gc
from glob import glob
import json
import numpy as np
import torch
#import kss
from time import time

from utils.data_loader import TextLoader
from others.utils import clean
from backbone.model_builder import BertSeparator
import IPython

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_json(p, lower=True):
    '''
    Presumm had this function load tgt tokens, but not mine.
    '''
    source = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if lower:
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if not flag:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    return source


def read_story(path):
    with open(path, 'r') as r:
        doc = r.readlines()
    return doc

def json_to_bert():
    files = glob(os.path.join('simple_inference', '*.story'))
    #article = kss.split_sentences(read_story(files[0])[0])
    article = read_story(files[0])
    article = [sent.strip() for sent in article if len(sent) >= 20]
    #line_data = [' '.join(sent) for doc in lines for sent in doc]
    #article = [sent for sent in article if len(sent) >= 20]
    return article

    
def get_model_params(checkpoint):
    global args
    opts = checkpoint['opt']
    args.add_transformer = opts.add_transformer
    args.classifier_type = opts.classifier_type
    args.window_size = opts.window_size


class SepInference:
    def __init__(self, args, model, device):
        self.device = device
        self.args = args
        self.text_loader = TextLoader(args, device)
        self.model = model.eval()

    def make_eval(self, doc):
        device = self.device
        args = self.args
        ws = args.window_size

        cands = ['\n'.join(doc[i:i+ws*2]) for i in range(len(doc) - ws*2 + 1)]
        tmp_batch = self.text_loader.load_text(cands)

        scores = np.zeros(len(doc) - 1)
        offset = ws - 1

        # caculate scores
        start = time()
        logits = []
        for i, batch in enumerate(tmp_batch):
            (src, segs, clss, mask_src, mask_cls), _ = batch

            assert clss.shape[-1] == ws*2
            logit = self.model(src, segs, clss, mask_src, mask_cls).detach().to('cpu').item()
            logits.append(logit)
            
        print(f"Elapsed Time: {time() - start}")

        tmp_batch, cands = [], []
        gc.collect()
        
        logits = np.array(logits)
        scores[offset:len(scores) - offset] = logits

        self.print_result(doc, scores)
    
    def print_result(self, doc, scores):
        threshold = self.args.threshold
        if os.path.exists('simple_inference/inference_result.txt'):
            os.remove('simple_inference/inference_result.txt')

        to_print = [0] * (len(doc) * 2 - 1)
        to_print[::2] = list(doc)
        #to_print[1::2] = [f'------ SEP {s:.2f} ------' if s > threshold else None for s in scores]
        to_print[1::2] = [f'------ SEP {s:.2f} ------' for s in scores]
        to_print = [line for line in to_print if line is not None]

        with open('simple_inference/inference_result.txt', 'a') as file:
            file.write('\n'.join(to_print))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("--backbone_type", default='bert', type=str, choices=['bert', 'bertsum'])
    parser.add_argument('--classifier_type', default='conv', type=str, choices=['conv', 'linear'])
    parser.add_argument("--mode", default='test', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("--random_seed", default=227182, type=int)

    #parser.add_argument("--bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("--dataset_path", default='dataset/')
    parser.add_argument("--model_path", default='models/')
    parser.add_argument("--result_path", default='results')
    parser.add_argument("--temp_dir", default='temp/')

    # dataset type
    parser.add_argument("--data_type", default='cnndm', type=str)
    parser.add_argument("--window_size", default=3, type=int)
    parser.add_argument("--y_ratio", default=0.5, type=float)
    parser.add_argument("--use_stair", action='store_true')
    parser.add_argument("--random_point", action='store_true')

    parser.add_argument("--batch_size", default=3000, type=int)
    parser.add_argument("--test_batch_size", default=200, type=int)

    parser.add_argument("--max_pos", default=512, type=int)
    parser.add_argument("--use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--load_from_extractive", default='', type=str)

    parser.add_argument("--sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--lr_bert", default=2e-3, type=float)
    parser.add_argument("--lr_dec", default=2e-3, type=float)
    parser.add_argument("--use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("--share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--finetune_bert", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("--dec_dropout", default=0.2, type=float)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dec_hidden_size", default=768, type=int)
    parser.add_argument("--dec_heads", default=8, type=int)
    parser.add_argument("--dec_ff_size", default=2048, type=int)
    parser.add_argument("--enc_hidden_size", default=512, type=int)
    parser.add_argument("--enc_ff_size", default=512, type=int)
    parser.add_argument("--enc_dropout", default=0.2, type=float)
    parser.add_argument("--enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument('--add_transformer', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--ext_dropout", default=0.2, type=float)
    parser.add_argument("--ext_layers", default=2, type=int)
    parser.add_argument("--ext_hidden_size", default=768, type=int)
    parser.add_argument("--ext_heads", default=8, type=int)
    parser.add_argument("--ext_ff_size", default=2048, type=int)

    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--generator_shard_size", default=32, type=int)
    parser.add_argument("--alpha",  default=0.6, type=float)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--min_length", default=15, type=int)
    parser.add_argument("--max_length", default=150, type=int)
    parser.add_argument("--max_tgt_len", default=140, type=int)

    parser.add_argument("--visible_gpus", default='1', type=str)
    parser.add_argument("--gpu_ranks", default='0', type=str)
    parser.add_argument("--log_dir", default='logs/traineval')

    # Eval
    parser.add_argument("--test_from", default='models/model_w3_fixed_step_50000.pt')
    parser.add_argument("--threshold", default=0.0, type=float)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    
    device = 'cuda'
    #device = 'cpu'
    device_id = 0

    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    get_model_params(checkpoint) # get arguments from saved opts
    
    model = BertSeparator(args, device_id, checkpoint).to(device)
    #model.to('cpu')

    inferencer = SepInference(args, model, device)
    doc = json_to_bert()

    inferencer.make_eval(doc=doc)
