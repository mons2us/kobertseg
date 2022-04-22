
import IPython

import os

import numpy as np
import torch
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter

import distributed
from utils.data_loader import DataLoader, load_dataset, TextLoader
from utils.reporter import ReportMgr, Statistics
from others.logging import logger


def csv_writer(args, step, acc_value, avg_sep, threshold):
    #model_index = args.test_from.split('/')[1].split('_')[-1]
    save_dir = '/'.join(args.test_from.split('/')[:2])
    #save_dir = os.path.join(args.model_path, f'index_{model_index}')
    file_path = os.path.join(save_dir, f'sep_result_{args.data_type}.csv')
    if not os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write("step,threshold,accuracy,avg_points\n")
    with open(file_path, 'a') as f:
        f.write(f"{step},{threshold},{acc_value},{avg_sep}\n")


def build_evaluator(args, device_id, model):
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=None)
    evaluator = Evaluator(args, model, n_gpu, gpu_rank, report_manager=report_manager)
    return evaluator


def build_sep_evaluator(args, device_id, model):
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=None)
    evaluator = SepEvaluator(args, model, device_id)
    return evaluator


class Evaluator:
    def __init__(self, args, model, n_gpu=1, gpu_rank=1, report_manager=None):
        self.args = args
        self.model = model
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        # to eval mode
        self.model.eval()

    def cls_eval(self, test_iter_fct=None):
        assert test_iter_fct
        logger.info("Evaluation: Classification Starts.")
        args = self.args
        test_iter = test_iter_fct()

        test_stats = Statistics()
        test_norm = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter)):
                test_norm += batch.batch_size
                src = batch.src
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sep_label = batch.sep_label

                sep_pred = self.model(src, segs, clss, mask, mask_cls)
                test_loss = self.loss(sep_pred, sep_label.float())
                test_n_correct = (sep_label == (sep_pred > self.args.threshold)).sum().to('cpu').item()
                
                test_batch_stats = Statistics(float(test_loss.detach().to('cpu').numpy()), test_norm, test_n_correct)
                test_stats.update(test_batch_stats)

                # initializae normalization denominator
                test_norm = 0
            self._report_step(0, 0, valid_stats=test_stats)

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats)


# !!TODO!! 문서가 3개 이상인 경우도 추가
class SepEvaluator:
    '''
    NLP tokenizing이 된 text를 input으로 사용
    해당 text 두개(7~10 + 7~10)를 붙여서 window size대로 슬라이딩을 하면서 bert tokenizing 수행
    '''
    def __init__(self, args, model, device):
        self.args = args
        self.text_loader = TextLoader(args, device)
        self.model = model
        self.device = device

        self.tot_nums = []
        self.tot_gt = []

    def _separate(self, scores, max_mode='max_all'):
        scores = torch.sigmoid(scores)
        if max_mode == 'max_all':
            sep_point = torch.argmax(scores) if max(scores) > self.args.threshold else -1e9
            return sep_point, None
        elif max_mode == 'max_one':
            sep_point = torch.where((scores > self.args.threshold) & (scores != 0.5))[0]
            count_max = len(sep_point)
            sep_point = -1e9 if len(sep_point) != 1 else sep_point
            return sep_point, count_max

    def sep_eval(self):
        args = self.args
        max_mode = args.test_max_mode

        # path to save result in .txt file
        setattr(self, 'save_pth', 'sep_result.txt')
        if os.path.exists(self.save_pth):
            os.remove(self.save_pth)

        ws = self.args.window_size
        mixed_doc_set = self.load_dataset()

        test_iter = DataLoader(args, mixed_doc_set, args.batch_size, 'cuda', False, True)
        self.model.eval()

        pred_result = torch.tensor([]).to(self.device)
        for i, batch in enumerate(tqdm(test_iter)):
            src = batch.src
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            sep_label = batch.sep_label

            sep_pred = self.model(src, segs, clss, mask, mask_cls)
            pred_result = torch.cat((pred_result, sep_pred.detach()))
            pred_result = pred_result.detach()

        self.batch_eval(pred_result)

    def load_dataset(self):
        args = self.args
        ws = args.window_size

        #dataset = [d['src_txt'] for d in torch.load('dataset/cnndm/bert_data/test_dataset.pt')]
        dt_pth = f'dataset/{args.data_type}/bert_data/sep_eval_dataset.pt'
        logger.info(f"Loading sep_eval data from {dt_pth}")
        dataset = torch.load(dt_pth)
        dataset = self._make_sepdata(dataset)

        # 각 dataset에서 모든 후보군 추출
        tot_cands, tot_nums, tot_gt = [], [], []
        for doc, label in dataset:
            cands = ['\n'.join(doc[i:i+ws*2]) for i in range(len(doc) - ws*2 + 1)]
            tot_cands.extend(cands)
            tot_nums.append(len(cands))
            tot_gt.append(label)
        
        tmp_batch = self.text_loader.load_text(tot_cands)
        setattr(self, 'tot_nums', tot_nums)
        setattr(self, 'tot_gt', tot_gt)
        
        batch_fin = []
        for i, batch in enumerate(tmp_batch):
            (src, segs, clss, mask_src, mask_cls), _ = batch
            batch_fin.append({'src': src[0].tolist(),
                            'segs': segs[0].tolist(),
                            'clss': clss[0].tolist(),
                            'mask_src': mask_src[0].tolist(),
                            'mask_cls': mask_cls[0].tolist()})
        yield batch_fin

    def _make_sepdata(self, dataset):
        '''
        generate mixed documents for given list of cleaned data
        '''
        args = self.args
        ws = args.window_size

        mixed_doc_set = []
        assert args.test_sep_num < len(dataset)
        max_num = len(dataset) - 1 if args.test_sep_num == -1 else args.test_sep_num

        for i in range(max_num):
            '''
            # of possible sep points: Total Length - (window size * 2) + 1 (e.g. 15 when tot_length = 20 and ws = 3)
            To make sep accuracy per window size balanced on its evaluation,
            length of a mixed article is set as;
                Total Length: 20 + (ws * 2)
                therefore the # of sep points are the same for all possible window sizes (= 21 for most cases)
                articles with less length than tot_len are just added so.
            '''
            
            # lh_count = min(random.randint(7, 10), len(dataset[i]))
            # rh_count = min(random.randint(7, 10), len(dataset[i+1]))
            # lh_doc = dataset[i][:lh_count]
            # rh_doc = dataset[i+1][:rh_count]
            # gt = lh_count - 1
            
            side_len = 7 + ws
            lh_count = min(side_len, len(dataset[i]))
            rh_count = min(side_len, len(dataset[i+1]))

            lh_doc = dataset[i][:lh_count]
            rh_doc = dataset[i+1][:rh_count]
            gt = lh_count - 1

            src_doc = lh_doc + rh_doc
            mixed_doc_set.append((src_doc, gt))
        
        return mixed_doc_set
    
    def batch_eval(self, pred_result):
        max_mode = self.args.test_max_mode
        gt = [(g - self.args.window_size + 1) for g in self.tot_gt]
        nums = self.tot_nums
        
        pointer = 0
        acc_cnt, err_cnt = 0, 0
        count_max = 0
        
        assert sum([n for n in nums]) == pred_result.size(0)
        for i, p in enumerate(nums):
            scores = pred_result[pointer:pointer+p]
            pointer += p
            split, tmp_count_max =  self._separate(scores, max_mode=max_mode)
            count_max += tmp_count_max if tmp_count_max else 0

            if split == gt[i]:
                acc_cnt += 1
                #self.print_result(self.save_pth, d, scores, True)
            else:
                err_cnt += 1

        acc = acc_cnt/(acc_cnt + err_cnt) * 100
        avg_sep = count_max / (acc_cnt + err_cnt)

        if max_mode == 'max_all':
            print(f"Evaluation Result: {acc:.2f}%")
        elif max_mode == 'max_one':
            print(f"Evaluation Result: {acc:.2f}%  Average Sep Points: {avg_sep:.2f}")
            
        # write into .csv file
        csv_writer(self.args, self.args.test_from, acc, avg_sep, self.args.threshold)

        # acc_cnt, err_cnt = 0, 0
        # count_max = 0
        # for idx, (d, _gt) in tqdm(enumerate(mixed_doc_set), total=len(mixed_doc_set)):
        #     scores = np.zeros(len(d) - 1)
        #     offset = ws - 1
        #     tmp_scores = self.eval_single_doc(d, _gt)
        #     scores[offset:len(scores) - offset] = tmp_scores

        #     pred_result, tmp_count_max = _separate(scores, max_mode)
        #     count_max += tmp_count_max if tmp_count_max else 0

        #     if pred_result == _gt:
        #         acc_cnt += 1
        #         self.print_result(save_pth, d, scores, True)
        #     else:
        #         err_cnt += 1
        #         self.print_result(save_pth, d, scores, False)

        # acc = acc_cnt/(acc_cnt+err_cnt)*100
        # avg_sep = count_max/(acc_cnt+err_cnt)

        # if max_mode == 'max_all':
        #     print(f"Evaluation Result: {acc:.2f}%")
        # elif max_mode == 'max_one':
        #     print(f"Evaluation Result: {acc:.2f}%  Average Sep Points: {avg_sep:.2f}")
        
        # # write into .csv file
        # csv_writer(args, args.test_from, acc, avg_sep)

    def eval_single_doc(self, doc, gt):
        '''
        for a given mixed document,
        calculate sentence by sentence logit score to decide separation points.
        '''

    def print_result(self, pth, doc, scores, flag):
        to_print = [0] * (len(doc) * 2 - 1)
        to_print[::2] = list(doc)
        to_print[1::2] = list(map(str, scores))

        prob_crit = 0.1 if flag else 0.05
        if random.random() > prob_crit:
            flag_text = '[CORRECT PREDICTION]\n' if flag else '[WRONG PREDICTION]\n'
            with open(pth, 'a') as file:
                file.write(flag_text)
                file.write('\n'.join(to_print))
                file.write('\n'*4)