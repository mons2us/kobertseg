import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from others.logging import logger
#from decoder import TransformerDecoder
from backbone.bertsep import Classifier, LinearClassifier, SepTransformerEncoder
from backbone.optimizers import Optimizer

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.utils import download as _download
from kobert.pytorch_kobert import get_pytorch_kobert_model


def get_kobert_vocab(cachedir="./tmp/"):
    # Add BOS,EOS vocab
    vocab_info = {
        'url': 'https://kobert.blob.core.windows.net/models/kobert/tokenizer/kobert_news_wiki_ko_cased-ae5711deb3.spiece',
        'fname': 'kobert_news_wiki_ko_cased-1087f8699e.spiece',
        'chksum': 'ae5711deb3'
    }
    
    vocab_file = _download(
        vocab_info["url"], vocab_info["chksum"], cachedir=cachedir
    )

    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_file, padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )

    return vocab_b_obj



def build_optim(args, model, checkpoint):
    """ Build optimizer """
    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)
    optim.set_parameters(list(model.named_parameters()))

    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)
    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """
    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)
    return optim


# def get_generator(vocab_size, dec_hidden_size, device):
#     gen_func = nn.LogSoftmax(dim=-1)
#     generator = nn.Sequential(
#         nn.Linear(dec_hidden_size, vocab_size),
#         gen_func
#     )
#     generator.to(device)

#     return generator

class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        temp_dir = args.temp_dir

        if args.large:
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            logger.info("Loading pretrained kobert model.")
            self.model, vocab = get_pytorch_kobert_model(cachedir=".cache")
            # add [BOS], [EOS]
            self.model.resize_token_embeddings(len(vocab)) 
            #self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        
        # Load bertsum weight
        # if mode is test or backbone used is BERT, there's no need to load bertsum weights
        if args.mode == 'train' and args.backbone_type == 'bertsum':
            logger.info("Using BertSum weights ---> loading the weights")
            self._load_bertsum_weight()
        else:
            logger.info("Not Using BertSum weights")
            
        # whether finetune the backbone (bert or bertsum)
        self.finetune = True if (args.finetune_bert) and (args.mode == 'train') else False
        if args.mode == 'train':
            if self.finetune:
                logger.info(f"Finetuning {args.backbone_type}")
            else:
                logger.info(f"Not finetuning backbone({args.backbone_type})")

    def _load_bertsum_weight(self):
        bertsum_weight = torch.load('models/model_step_130000.pt')
        bert_dict = self.model.state_dict()
        print("before:", bert_dict['encoder.layer.0.attention.self.query.weight'])
        bertsum_dict = bertsum_weight['model']

        # Filter out ext_layer weights, which do not exist in pretrained bert model
        bertsum_dict = {k.split('bert.model.')[-1]: v for k, v in bertsum_dict.items() if k.split('bert.model.')[-1] in bert_dict}
        bert_dict.update(bertsum_dict)
        self.model.load_state_dict(bert_dict)
        print("BertSum weights loaded.")
        print("after:", self.model.state_dict()['encoder.layer.0.attention.self.query.weight'])
        return

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class BertSeparator(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(BertSeparator, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args)
        
        if args.add_transformer:
            logger.info(f"Using Transformer Layer: {args.backbone_type} --> [transformer * {args.ext_layers}] --> classifier")
            self.sep_layer = SepTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads, args.ext_dropout, args.ext_layers)
        else:
            logger.info(f"Not Using Transformer Layer: {args.backbone_type} --> (no transformer) --> classifier")

        if args.classifier_type == 'conv':
            self.classifier = Classifier(args.window_size)
        else:
            self.classifier = LinearClassifier(args.window_size)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
            logger.info("Checkpoint loaded on the model.")
        else:
            if args.add_transformer:
                if args.param_init != 0.0:
                    for p in self.sep_layer.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                if args.param_init_glorot:
                    for p in self.sep_layer.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)
        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        if self.args.add_transformer:
            sents_vec = self.sep_layer(sents_vec, mask_cls).squeeze(-1)
        classified = self.classifier(sents_vec)
        return classified

# class AbsSummarizer(nn.Module):
#     def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
#         super(AbsSummarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

#         if bert_from_extractive is not None:
#             self.bert.model.load_state_dict(
#                 dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
#                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
#                                     intermediate_size=args.enc_ff_size,
#                                     hidden_dropout_prob=args.enc_dropout,
#                                     attention_probs_dropout_prob=args.enc_dropout)
#             self.bert.model = BertModel(bert_config)

#         if (args.max_pos > 512):
#             my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#             my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#             my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#             self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#         self.vocab_size = self.bert.model.config.vocab_size
#         tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#         if (self.args.share_emb):
#             tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

#         self.decoder = TransformerDecoder(
#             self.args.dec_layers,
#             self.args.dec_hidden_size, heads=self.args.dec_heads,
#             d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

#         self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
#         self.generator[0].weight = self.decoder.embeddings.weight


#         if checkpoint is not None:
#             self.load_state_dict(checkpoint['model'], strict=True)
#         else:
#             for module in self.decoder.modules():
#                 if isinstance(module, (nn.Linear, nn.Embedding)):
#                     module.weight.data.normal_(mean=0.0, std=0.02)
#                 elif isinstance(module, nn.LayerNorm):
#                     module.bias.data.zero_()
#                     module.weight.data.fill_(1.0)
#                 if isinstance(module, nn.Linear) and module.bias is not None:
#                     module.bias.data.zero_()
#             for p in self.generator.parameters():
#                 if p.dim() > 1:
#                     xavier_uniform_(p)
#                 else:
#                     p.data.zero_()
#             if(args.use_bert_emb):
#                 tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#                 tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#                 self.decoder.embeddings = tgt_embeddings
#                 self.generator[0].weight = self.decoder.embeddings.weight

#         self.to(device)

#     def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
#         top_vec = self.bert(src, segs, mask_src)
#         dec_state = self.decoder.init_decoder_state(src, top_vec)
#         decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
#         return decoder_outputs, None
