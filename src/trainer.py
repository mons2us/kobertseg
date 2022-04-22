import os

import numpy as np
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import distributed
from utils.reporter import ReportMgr, Statistics
from others.logging import logger
#from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    grad_accum_count = args.accum_count
    n_gpu = args.world_size # number of available gpus

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print(f'gpu_ranks: {args.gpu_ranks}')
    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = os.path.join(args.model_path, f'index_{args.model_index}')
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print total number of params
    if model:
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
                training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
                training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
                the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim, grad_accum_count=1, n_gpu=1, gpu_rank=1, report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        assert grad_accum_count > 0

        # Set model in training mode.
        if model:
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info("Start training...")
        args = self.args

        # Set save directory
        save_dir = os.path.join(self.args.model_path, f'index_{self.args.model_index}')
        # if not os.path.exists(save_dir):
        #     logger.info(f"Making save directory at: {save_dir}")
        #     os.makedirs(save_dir)
        # else:
        #     logger.info(f"Model path: {save_dir} already exists. Please check before training new model.")
        setattr(self, 'save_dir', save_dir)

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        #valid_len = len([d for d in valid_iter_fct()]) # takes time, but for tqdm
        valid_iter = valid_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                '''
                하나의 배치는 아래와 같은 형태
                tensor([[  101, 10587,  4783,  ...,   102,     0,     0],
                        [  101,  1006, 13229,  ...,   102,     0,     0],
                        [  101,  2385, 29627,  ...,   102,     0,     0],
                        [  101,  1037,  2148,  ..., 16540,   102,     0],
                        [  101,  2385, 29627,  ...,  5270,   102,     0],
                        [  101,  1006, 13229,  ...,  8853,  1012,   102]], device='cuda:0') 
                일반적으로는 [5, 512(max_pos)]의 형태를 따르나 길이에 따라 [10, 239] 등의 형태도 존재
                '''
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1

                        if self.n_gpu > 1:
                            normalization = sum(distributed.all_gather_list(normalization))

                        # For an iteration of training,
                        # do train -> update statistics (report manager) 
                        self._gradient_accumulation(true_batchs, normalization, total_stats, report_stats)
                        
                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break

                # Validation
                if (step % args.valid_steps == 0):
                    self.validate(valid_iter=valid_iter, step=step)
                    valid_iter = valid_iter_fct()

            train_iter = train_iter_fct()
        return total_stats

    def validate(self, valid_iter=None, step=0):
        logger.info("Start Validation")
        self.model.eval()
        val_stats = Statistics()
        val_norm = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_iter)):
                val_norm += batch.batch_size
                src = batch.src
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sep_label = batch.sep_label

                sep_pred = self.model(src, segs, clss, mask, mask_cls)
                val_loss = self.loss(sep_pred, sep_label.float())
                val_n_correct = (sep_label == (sep_pred > 0)).sum().to('cpu').item()
                
                val_batch_stats = Statistics(float(val_loss.detach().to('cpu').numpy()), val_norm, val_n_correct)
                val_stats.update(val_batch_stats)

                # initializae normalization denominator
                val_norm = 0
            self._report_step(0, step, valid_stats=val_stats)
        # to train mode
        self.model.train()


    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            sep_label = batch.sep_label

            sep_pred = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sep_pred, sep_label.float())
            n_correct = (sep_label == (sep_pred > 0)).sum().to('cpu').item()

            loss.backward()

            # TODO 이걸 해야되나?
            # loss.div(float(normalization)).backward()

            # Training process report (statistics)
            # normalization: Value kept added, every time a new batch comes in
            batch_stats = Statistics(float(loss.detach().to('cpu').numpy()), normalization, n_correct)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update after gradients are accumulated
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        random_flag = 'random' if self.args.random_point else 'fixed'
        checkpoint_path = os.path.join(self.save_dir, f'model_w{self.args.window_size}_{random_flag}_step_{str(step).zfill(5)}.pt')
        logger.info("Saving checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases
        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)
        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    # def test(self, test_iter, step, cal_lead=False, cal_oracle=False, return_results=False):
    #     """ Validate model.
    #         valid_iter: validate data iterator
    #     Returns:
    #         :obj:`nmt.Statistics`: validation loss statistics
    #     """
    #     # Set model in validating mode.
    #     def _get_ngrams(n, text):
    #         ngram_set = set()
    #         text_length = len(text)
    #         max_index_ngram_start = text_length - n
    #         for i in range(max_index_ngram_start + 1):
    #             ngram_set.add(tuple(text[i:i + n]))
    #         return ngram_set

    #     def _block_tri(c, p):
    #         tri_c = _get_ngrams(3, c.split())
    #         for s in p:
    #             tri_s = _get_ngrams(3, s.split())
    #             if len(tri_c.intersection(tri_s)) > 0:
    #                 return True
    #         return False

    #     if (not cal_lead and not cal_oracle):
    #         self.model.eval()
    #     stats = Statistics()
    #     oris = []
    #     can_path = '%s_step%d.candidate' % (self.args.result_path, step)
    #     gold_path = '%s_step%d.gold' % (self.args.result_path, step)
    #     with open(can_path, 'w') as save_pred:
    #         with open(gold_path, 'w') as save_gold:
    #             with torch.no_grad():
    #                 for batch in test_iter:
    #                     src = batch.src
    #                     if not return_results:
    #                         labels = batch.src_sent_labels
    #                     segs = batch.segs
    #                     clss = batch.clss
    #                     mask = batch.mask_src
    #                     mask_cls = batch.mask_cls

    #                     gold = []
    #                     pred = []
    #                     ori = [] # 원본 문장을 담을 list
    #                     idcs = []

    #                     if cal_lead:
    #                         selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
    #                     elif cal_oracle:
    #                         selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in range(batch.batch_size)]
    #                     else:
    #                         sents_vec, sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

    #                         if not return_results:
    #                             loss = self.loss(sent_scores, labels.float())
    #                             loss = (loss * mask.float()).sum()
    #                             batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
    #                             stats.update(batch_stats)
                                
    #                         '''
    #                         Token 제한 내 문장들에 대해 요약 score 저장 -> rank
    #                         '''
    #                         sent_scores = sent_scores + mask.float()
    #                         sent_scores = sent_scores.cpu().data.numpy()
    #                         selected_ids = np.argsort(-sent_scores, 1)
    #                     # selected_ids = np.sort(selected_ids,1)
    #                     '''
    #                     i: selected_ids가 하나 이상의 블록으로 나올 수 있는지는 모르겠으나, 그냥 결과물임
    #                     j: 결과물 중 실제로 
    #                     '''
    #                     for i, idx in enumerate(selected_ids):
    #                         _pred = []
    #                         _idcs = []
    #                         _ori  = []
    #                         if (len(batch.src_str[i]) == 0):
    #                             continue
    #                         for j in selected_ids[i][:len(batch.src_str[i])]:
    #                             if (j >= len(batch.src_str[i])):
    #                                 continue
    #                             candidate = " ".join(batch.src_str[i][j]).strip()
    #                             candidate_ori = batch.src_ori[j]
                                
    #                             if (self.args.block_trigram):
    #                                 if not _block_tri(candidate, _pred):
    #                                     _pred.append(candidate)
    #                                     _idcs.append(j)
    #                                     _ori.append(candidate_ori)
    #                             else:
    #                                 _pred.append(candidate)
    #                                 _idcs.append(j)
    #                                 _ori.append(candidate_ori)
                                    

    #                             if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
    #                                 break

    #                         _pred = " ".join(_pred).replace(" ", "")
    #                         _ori = " ".join(_ori)
    #                         if (self.args.recall_eval):
    #                             _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

    #                         pred.append(_pred)
    #                         idcs.append(_idcs)
    #                         ori.append(_ori)
    #                         gold.append(batch.tgt_str[i])

    #                     for i in range(len(gold)):
    #                         save_gold.write(gold[i].strip() + '\n')
    #                     for i in range(len(pred)):
    #                         save_pred.write(pred[i].strip() + '\n')
    #                     oris.append(ori[i])
                        
    #                 if return_results:
    #                     return sents_vec, oris, idcs
                            
    #     if (step != -1 and self.args.report_rouge):
    #         rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
    #         logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
    #     self._report_step(0, step, valid_stats=stats)
    #     return stats
