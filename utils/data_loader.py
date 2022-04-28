import bisect
import gc
import os
import glob
import random
import torch

import IPython

from others.logging import logger
from others.tokenization import BertTokenizer

import gluonnlp as nlp
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
#from kobert.utils import get_tokenizer
#from kobert.utils import download as _download
#from kobert.pytorch_kobert import get_pytorch_kobert_model


# def get_kobert_vocab(cachedir="./tmp/"):
#     # Add BOS,EOS vocab
#     vocab_info = {
#         'url': 'https://kobert.blob.core.windows.net/models/kobert/tokenizer/kobert_news_wiki_ko_cased-ae5711deb3.spiece',
#         'fname': 'kobert_news_wiki_ko_cased-1087f8699e.spiece',
#         'chksum': 'ae5711deb3'
#     }
    
#     vocab_file = _download(
#         vocab_info["url"], vocab_info["chksum"], cachedir=cachedir
#     )

#     vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
#         vocab_file, padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
#     )

#     return vocab_b_obj


class Batch:
    '''
    input_data
        data_dict = {'src': src_subtoken_idxs,
                    'segs': segments_ids,
                    'clss': cls_ids,
                    'src_txt': src_txt,
                    'sep_label': lab_idx}
    '''
    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_segs = [x[1] for x in data]
            pre_clss = [x[2] for x in data]
            src_str = [x[3] for x in data]
            sep_label = [x[4] for x in data]
            
            # !!padding token is 1 for kor!!
            src = torch.tensor(self._pad(pre_src, 1))
            segs = torch.tensor(self._pad(pre_segs, 1))
            mask_src = ~(src == 1)

            clss = torch.tensor(self._pad(pre_clss, -1))
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            sep_label = torch.tensor(sep_label)

            setattr(self, 'src', src.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'sep_label', sep_label.to(device))

            setattr(self, 'src_str', src_str)
            

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    '''
    미리 저장해 둔 mixed/normal dataset을 불러오는 function
    Args
        corpus_type: 'train', 'valid', 'test'
    Returns
        Every document of the specified file in Bert-data format
    '''
    assert corpus_type in ['train', 'valid', 'test']

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    ws = args.window_size
    
    # !!TODO!! Add stair arguments: from dataset generation stage!!
    stair_flag = 'stair' if args.use_stair else 'nostair'
    
    random_flag = 'random' if args.random_point else 'fixed'
    pt = os.path.join(args.dataset_path, f'{args.data_type}/kobertseg_dataset/kobertseg_dt_{corpus_type}_w{args.window_size}.pt')
    yield _lazy_dataset_loader(pt, corpus_type)

def batch_size_fn(new, count):
    '''
    Batch size function for token length
    Returns max size of sources(tokens) that can be in the batch.
    '''
    src = new[0]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class DataLoader:
    def __init__(self, args, datasets, batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator:
    '''
    input data
        data_dict = {'src': src_subtoken_idxs,
                    'segs': segments_ids,
                    'clss': cls_ids,
                    'src_txt': src_txt,
                    'sep_label': lab_idx}
    '''
    def __init__(self, args, dataset, batch_size, device=None, is_test=False, shuffle=True):
        self.args = args
        self.device = device
        self.batch_size = batch_size
        self.is_test = is_test
        self.dataset = dataset
        self.iterations = 0
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])
        self._iterations_this_epoch = 0
        self.batch_size_fn = batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        args = self.args
        src = ex['src']
        segs = ex['segs']
        clss = ex['clss']
        
        # separation evaluation인 경우
        if is_test:
            return src, segs, clss, '', 0

        src_txt = ex['src_txt']
        sep_label = ex['sep_label']
        
        end_id = [src[-1]] # [102]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        clss = clss[:max_sent_id]

        # If length of ex(=number of sentences) is less than window_size * 2,
        # do not add to minibatch
        if len(clss) != args.window_size * 2:
            return None
        
        # else return as normal dataset
        return src, segs, clss, src_txt, sep_label

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)

            if not ex:
                continue

            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch
    
    def batch(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = self.batch(p_batch, self.batch_size) # ?
            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b
    
    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)
                yield batch
            return


class TextLoader:
    '''
    Load text dataset(normal documents) into embeded format.
    It takes as init param the 'window size', which later results in (window_size * 2)*dim embedding
    '''
    def __init__(self, args, device):
        _, self.vocab = get_pytorch_kobert_model(cachedir=".cache")
        self.tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), self.vocab, lower=False)
        self.device = device
        self.args = args

    def load_text(self, source):
        '''
        source: input of two articles synthesized or one single article;
        It trims the source sentences if needed; From the longest
        '''
        args = self.args
        device = self.device

        sep_vid = self.vocab.token_to_idx['[SEP]']
        cls_vid = self.vocab.token_to_idx['[CLS]']
        pad_idx = self.vocab["[PAD]"]

        source = source if isinstance(source, list) else [source]

        def _process_src(raw):
            raw = raw.strip().lower()
            raw = raw.replace('[cls]','[CLS]').replace('[sep]','[SEP]')
            
            # Calculate average number of tokens per sentence,
            # for each case when using Bert or BertSum
            avg_token_num = self._get_avg_token_num(args)
            src_tokens = [self.tokenizer(r)[:avg_token_num] for r in raw.split('\n')]

            # Subtokenize
            src_subtokens = [['[CLS]'] + _set + ['[SEP]'] for _set in src_tokens]
            
            raw_original = raw.split('\n')
            raw = ' '.join([' '.join(s) for s in src_subtokens])
            src_subtokens = [x for y in src_subtokens for x in y]

            src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
            src_subtoken_idxs = src_subtoken_idxs[:-1][:args.max_pos]
            src_subtoken_idxs[-1] = sep_vid
            
            _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
            segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
            segments_ids = []
            segs = segs[:args.max_pos]
            for i, s in enumerate(segs):
                if (i % 2 == 0):
                    segments_ids += s * [0]
                else:
                    segments_ids += s * [1]

            src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
            mask_src = (~(src == 1)).to(device)
            cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
            clss = torch.tensor(cls_ids).to(device)
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            return raw, raw_original, src, mask_src, segments_ids, clss, mask_cls

        for x in source:
            x, x_org, src, mask_src, segments_ids, clss, mask_cls = _process_src(x) ## !!Edit!!
            segs = torch.tensor(segments_ids)[None, :].to(device)
            batch = src, segs, clss, mask_src, mask_cls
            yield batch, x_org

    def _get_avg_token_num(self, args):
        sent_len = args.window_size * 2
        if args.backbone_type == 'bert':
            avg_token_num = (args.max_pos - sent_len - 1) // sent_len
        elif args.backbone_type == 'bertsum':
            avg_token_num = (args.max_pos - sent_len*2) // sent_len
        return avg_token_num