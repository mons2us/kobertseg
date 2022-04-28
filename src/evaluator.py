
import IPython

import os
import itertools
import numpy as np
import torch
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter
import segeval
from sklearn.metrics import precision_score, recall_score, f1_score

import distributed
from utils.data_loader import DataLoader, load_dataset, TextLoader
from utils.reporter import ReportMgr, Statistics
from others.logging import logger



def csv_writer(args, step, precision, recall, f1_score, pk, wd, threshold):
    #model_index = args.test_from.split('/')[1].split('_')[-1]
    save_dir = '/'.join(args.test_from.split('/')[:2])
    #save_dir = os.path.join(args.model_path, f'index_{model_index}')
    file_path = os.path.join(save_dir, f'sep_result_{args.data_type}.csv')
    if not os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write("step,threshold,precision,recall,f1_score,pk,wd\n")
    with open(file_path, 'a') as f:
        f.write(f"{step},{threshold},{precision},{recall},{f1_score},{pk},{wd}\n")


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
        
        # 미리 합성된 기사를 불러오도록 함
        mixed_doc_set = self.load_dataset()

        test_iter = DataLoader(args, mixed_doc_set, args.batch_size, 'cuda', False, True)
        self.model.eval()

        pred_result = torch.tensor([])
        for i, batch in enumerate(tqdm(test_iter)):
            src = batch.src
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls
            sep_label = batch.sep_label

            sep_pred = self.model(src, segs, clss, mask, mask_cls)
            pred_result = torch.cat((pred_result, sep_pred.detach().to('cpu')))
            #pred_result = pred_result.detach().to('cpu')

        self.batch_eval(pred_result)

    def load_dataset(self):
        args = self.args
        ws = args.window_size

        #dataset = [d['src_txt'] for d in torch.load('dataset/cnndm/bert_data/test_dataset.pt')]
        dt_pth = f'dataset/{args.data_type}/eval_data/eval_set.pt'
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
        
        setattr(self, 'tot_nums', tot_nums)
        setattr(self, 'tot_gt', tot_gt)
        #tmp_batch = self.text_loader.load_text(tot_cands)

        def _lazy_batch_loader(tot_cands):
            tmp_batch = self.text_loader.load_text(tot_cands)
            batch_fin = []
            for i, batch in enumerate(tmp_batch):
                (src, segs, clss, mask_src, mask_cls), _ = batch
                batch_fin.append({'src': src[0].tolist(),
                                'segs': segs[0].tolist(),
                                'clss': clss[0].tolist(),
                                'mask_src': mask_src[0].tolist(),
                                'mask_cls': mask_cls[0].tolist()})
            return batch_fin

        # batch_fin = []
        # for i, batch in enumerate(tmp_batch):
        #     (src, segs, clss, mask_src, mask_cls), _ = batch
        #     batch_fin.append({'src': src[0].tolist(),
        #                     'segs': segs[0].tolist(),
        #                     'clss': clss[0].tolist(),
        #                     'mask_src': mask_src[0].tolist(),
        #                     'mask_cls': mask_cls[0].tolist()})
        yield _lazy_batch_loader(tot_cands)
    
    def _make_sepdata(self, dataset):
        k = self.args.window_size - 1
        if k == 0:
            return dataset
        else:
            res_dataset = []
            for doc, label in dataset:
                doc = doc[:k] + doc + doc[-k:]
                res_dataset.append((doc, label))
            return res_dataset

    def batch_eval(self, pred_result):
        args = self.args
        wd, pk, precision, recall, fone = [], [], [], [], []

        pointer = 0
        for i, num in enumerate(self.tot_nums):
            tmp_pred = pred_result[pointer:pointer + num]
            
            # for window comparision
            # max window: 5
            if args.compare_window:
                tmp_pred = tmp_pred[4:-4]
                pred_num = num - 7
            else:
                pred_num = num + 1

            pred = torch.cat((
                torch.tensor([0]),
                torch.where(torch.sigmoid(tmp_pred) >= self.args.threshold)[0].detach().to('cpu') + 1,
                #torch.sort(tmp_doc.topk(4).indices).values.detach().to('cpu') + 1,
                torch.tensor([pred_num])
            ))

            pred_seg = [(pred[i+1] - pred[i]).item() for i in range(len(pred)-1)]
            #pred_seg = [n - 5 if i in [0, len(pred_seg) - 1] else n for i, n in enumerate(pred_seg)]
            
            ref_seg = self.tot_gt[i]
            # for comparison between windows
            # max window: 5
            if args.compare_window:
                ref_seg = [n - 4 if i in [0, len(ref_seg) - 1] else n for i, n in enumerate(ref_seg)]
            
            pred_bin = [[0]*(n-1) + [1] for n in pred_seg]
            pred_bin = list(itertools.chain(*pred_bin))[:-1]

            ref_bin = [[0]*(n-1) + [1] for n in ref_seg]
            ref_bin = list(itertools.chain(*ref_bin))[:-1]

            # pred_points = np.cumsum(pred_seg)[:-1] - 1
            # ref_points = np.cumsum(ref_seg)[:-1] - 1

            # pred_bin = [1 if i in pred_points else 0 for i in range(num)]
            # ref_bin = [1 if i in ref_points else 0 for i in range(num)]
            pre = precision_score(y_true=ref_bin, y_pred=pred_bin, average='binary')
            rec = recall_score(y_true=ref_bin, y_pred=pred_bin, average='binary')
            f1 = f1_score(y_true=ref_bin, y_pred=pred_bin, average='binary')

            precision.append(pre)
            recall.append(rec)
            fone.append(f1)
            wd.append(float(segeval.window_diff(ref_seg, pred_seg)))
            pk.append(float(segeval.pk(ref_seg, pred_seg)))

            pointer += num

        precision = np.mean(precision)
        recall = np.mean(recall)
        fone = np.mean(fone)
        pk = np.mean(pk)
        wd = np.mean(wd)

        # write into .csv file
        csv_writer(self.args, self.args.test_from, precision, recall, fone, pk, wd, self.args.threshold)



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
