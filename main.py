'''
Train and Evaluate sequential EBMs
'''
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import argparse
import sys
import os
import os.path as osp
from tqdm import tqdm
sys.path.append('/home/yichuan/HKU/EBM/ire_reasoning')
os.chdir('/home/yichuan/HKU/EBM/ire_reasoning')
# print(f'The current working directory: {os.getcwd()}')
import hydra
from sequential_ebms import BERTSequentialEBMs, GPTSequentialEBMs, FastSequentialEBMs
from dataset import load_data
from utils import convert_time, VisualizeEBMs
import random as rand
import wandb
from time import time
import json
import math

IGNORE_INDEX = -100

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]  # Remove "module." prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def test(model, test_data):
    '''
    trivial test to check the output of model.generate()
    '''
    #batchalize
    test_data = torch.nn.functional.one_hot(test_data.type(torch.long), model.num_classes)
    x = test_data[:, :model.inp_len, :]
    y = test_data[:, model.inp_len:, :]
    #tokenize
    for k in range(1, model.ebm_num+1):
        output_dict = model.generate(x, k)
        print(f'\n___________\n{k}-th output_dict: \n{output_dict}\n__________\n')
        
def unmasking_schedule(k, scheduler='cosine'):
    if scheduler == 'cosine':
        return 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
    elif scheduler == 'linear':
        return np.linspace(1, 0, k)
    elif scheduler == 'quadratic':
        return np.linspace(1, 0, k) ** 2
    else:
        raise NotImplementedError

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
    def fast_forward(self, n_steps: int):
        '''adjust the lr to the n-th step in one-run'''
        self.n_current_steps = n_steps
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-3, ema_beta=0.9, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.ema_beta = ema_beta
        self.mode = mode
        self.best = math.inf if mode == 'min' else -math.inf
        self.num_bad = 0
        self.ema = None

    def update(self, value):
        # EMA to tame vibration
        self.ema = value if self.ema is None else self.ema_beta*self.ema + (1-self.ema_beta)*value
        metric = self.ema

        improved = (metric < self.best - self.min_delta) if self.mode == 'min' else (metric > self.best + self.min_delta)
        if improved:
            self.best = metric
            self.num_bad = 0
            return False  # don't stop
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience

class SequentialEBMsTrainer:
    def __init__(
        self,
        sebm,
        train_dataloader,
        test_dataloader,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        sampler='gibbs',
        sampling_times=10,
        log_freq=10, #log every 10 batches
        train_size=-1,
        test_size=-1,
        train_wandb=True,
        test_wandb=False,
        is_ebm=True,
        contrast=False,
        device='cpu', ########debugging
        parallel=False,
        epochs=-1,
        continue_train=False,
    ):
        self.device = device
        self.parallel = parallel
        self.sebm = sebm
        self.is_ebm = is_ebm
        self.contrast = contrast
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.train_size=train_size
        self.test_size=test_size
        # Schedule the optimizer (as stated in paper)
        self.optim = optim.AdamW(self.sebm.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.sebm.d_model, n_warmup_steps=warmup_steps #d_model决定initial_lr
        )
        self.criterion = nn.CrossEntropyLoss().to(device) #label_smoothing=True
        self.log_freq = log_freq
        # sampling configs:
        self.sampler = sampler
        self.sampling_times = sampling_times
        self.train_wandb = train_wandb
        self.test_wandb = test_wandb
        self.epochs = epochs
        print(f"Total Parameters: {sum([p.nelement() for p in self.sebm.model.parameters()])}, is_ebm: {self.is_ebm}")
        self.ckpts_path = f'./ire_reasoning/ebm_ckpts/{self.sebm.task_name}_{self.sebm.param_type}{self.sebm.d_model}_earlystop.pth' # _ebm8.9 (sota with sampling_new())
        
        if continue_train:
            self.load_model(self.ckpts_path, device)
            print(f'\n\nContinue train from {self.ckpts_path}, ckpts loaded.')
        else:
            print(f'\n\nTrain from scratch')
    
    def load_model(self, ckpts_path, device): #config.device
        if isinstance(device, int):
            device = torch.device('cuda', device)
        state_dict = torch.load(ckpts_path, map_location=device, weights_only=True)
        state_dict = remove_module_prefix(state_dict)
        self.sebm.model.load_state_dict(state_dict)
        
    def train(self, epoch, stage):
        if self.sebm.param_type == 'fast':
            self.fast_iteration(epoch, self.train_data, stage, parallel=self.parallel)
        else:
            self.iteration(epoch, self.train_data, stage, is_ebm=self.is_ebm)

    def test(self, epoch, stage): #暂时无用
        self.iteration(epoch, self.test_data, stage, train=False)
        
    def validate(self, epoch, early_stopper):
        '''validate on the test set'''
        self.sebm.model.eval()
        val_loss = 0.0
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            t_list = unmasking_schedule(10+2, 'cosine')[1:-1]
            if self.sebm.param_type == 'fast':
                if (not self.parallel) or self.device == 0:
                    data_iter = data_iter = tqdm(
                        enumerate(self.test_data),
                        desc="EP_%s_%s:%d" % ('train', 'validate', epoch),
                        total=len(self.test_data),
                        bar_format="{l_bar}{r_bar}"
                    )
                else:
                    data_iter = enumerate(self.test_data)
            else:
                raise NotImplementedError('the param_type "{}" is currently not implemented!')
            if self.sebm.task_name == 'countdown':
                special_tokens = {0, 1, 2, 3, 4}
                self.sebm.special_tok_size = len(special_tokens)
            else:
                raise NotImplementedError
            
            for i, data in data_iter:
                B = data['label'].size(0)
                scheduled_data, K = self.add_schedule_new(data, t_list)
                partial_pred = scheduled_data['input'].clone()
                partial_pred[partial_pred == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
                for k in range(K):
                    data = self.prepare_partial_data(scheduled_data, k+1)
                    data = {k:v.to(self.device) for k,v in data.items()}
                    u = self.sebm.get_2D_indices(
                        array=data['label'],
                        val=self.sebm.tokenizer.pad_token_id,
                        type='remove_pad',
                    )
                    if k == K-1:
                        get_logits=True
                    else:
                        get_logits=False
                    partial_pred, sth = self.sebm.sampling_revised( #sth have to store the ce loss of the batch
                        partial_pred,
                        u,
                        self.sampler,
                        self.sampling_times,
                        batch_id=k,
                        get_logits=get_logits,
                    )
                # end of K iter
                xu_label = data['label'].gather(dim=1, index=u)
                ce_loss = self.criterion(sth['pseudo_xu_logits'], xu_label).item() #(B,V,U), (B,U) -> scalar
                val_loss += ce_loss * B
                total_samples += B
                pred = partial_pred
                pred, scheduled_data['label'] = pred.to(self.device), scheduled_data['label'].to(self.device)
                invalid_pos = scheduled_data['label'] < len(special_tokens)
                pred[invalid_pos] = scheduled_data['label'][invalid_pos]
                correct = self.eval_metric(pred, scheduled_data['label'])
                total_correct += correct
            # end of batch iter
            val_ce = val_loss / total_samples
            final_acc = round(total_correct*100/total_samples, 2)
            if early_stopper.update(val_ce):
                return True, final_acc, val_ce
            else:
                return False, final_acc, val_ce
    
    def evaluate(self, k, scheduler='cosine', stage='pretrain', visualize=False): #Inference
        '''
        Recover a fully masked sequence using a specified scheduler, with decreasing t
        '''
        t_list = unmasking_schedule(k+2, scheduler)[1:-1]
        print(f't_list: {t_list}')
        if self.sebm.param_type == 'fast':
            self.fast_iteration(1, self.test_data, stage, train=False, visualize=visualize)
        else:
            self.iteration(1, self.test_data, stage=stage, train=False, \
                schedule=t_list, parallel=self.parallel, visualize=visualize)
        
    def add_schedule(self, data, schedule):
        '''
        simply add a schedule label key-value pair to the data dict
        
        params:
            data: batchalized data dict with bert_inputs, bert_labels, segment_labels and contrast
            schedule: a list of decreasing masking rate t (fully masked to fully unmasked)
        return:
            scheduled_data: batchalized data dict extended with schedule_label k-v pair
        '''
        scheduled_data = data
        batch_size, io_len = data['label'].size()
        # print(f"bert_label({data['bert_label'].shape}): {data['bert_label']}")
        schedule_label = [[0 for c in range(io_len)] for r in range(batch_size)] #initialize a 2D batchalized schedule label (4,30)
        special_ids = {0,1,2,3,4} #custom_tokenizer: pad, sep, mask, eos, unk
        # special_ids = {0} #mask
        full_label = data['label'].tolist()
        assert len(full_label)==batch_size and len(full_label[0])==io_len, \
            f'label: Size({len(full_label)}, {len(full_label[0])})'
        iters_count = 0 #early stop, can be lesser than k
        for k, t in enumerate(schedule):
            iters_count += 1
            unmask_num = int((1-t) * torch.count_nonzero(data['label'][0]))
            if unmask_num == 0:
                unmask_num = 1
            # print(f'k: {k}, t: {t}, unmask_num: {unmask_num}')
            for bid in range(batch_size): #each row inside the batch shares the same sequence of umask num
                labeled_num = len([x for x in schedule_label[bid] if x != 0])
                unlabeled_pos = [pos for pos in range(len(schedule_label[bid])) \
                    if (schedule_label[bid][pos] == 0 and full_label[bid][pos]>=len(special_ids))] #full_label[bid][pos]>0
                sample_num = unmask_num - labeled_num
                unmask_pos = rand.sample(unlabeled_pos, min(len(unlabeled_pos), sample_num))
                for pos in unmask_pos:
                    schedule_label[bid][pos] = (k+1) #add order label (1-indexed)
            # print(f'after k-{k+1}th schedule, scheudle_label: \n{schedule_label}')
            # if len(unlabeled_pos) <= sample_num: #complete labeling #由于batch内每行input len 可能不同，这里不考虑early_stop
            unmasked_pos = 0
            for bid in range(batch_size):
                unmasked_pos += len([pos for pos in range(len(schedule_label[bid])) \
                    if (schedule_label[bid][pos] == 0 and full_label[bid][pos]>=len(special_ids))])
            if unmasked_pos == 0:
                break
            # assert unmask_num > 0, f'unmasking number should be positive! Got {unmask_num}'
            # for bid in range(batch_size):
            #     for pos, label in enumerate(full_label[bid]):
            #         #pick the remaining t2 tokens as unmasking candidate
            #         if label not in special_ids and schedule_label[bid][pos]==0: 
            #             prob = rand.random()
            #             if prob < t:
            #                 schedule_label[bid][pos] = k+1 #1-indexed
            #         if torch.count_nonzero(schedule_label[bid]) == unmask_num:
            #             break
        schedule_label = torch.tensor(schedule_label)
        # for pos in data['bert_input'].size(1): #add CLS and SEP tokens
        #     if data['bert_input'][:, pos][0] == 1 or data['bert_input'][:, pos][0] == 2:
        #         schedule_label[:, pos] = data['bert_input'][:, pos]
        # print(f'\nInside add_schedule(), schedule_label: \n{schedule_label}\n, bert_label: \n{data["bert_label"]}')
        scheduled_data['schedule_label'] = schedule_label#.to(self.device)
        # 以下assertion变了： torch.count_nonzero(data['bert_label'])
        valid_labels = (data['label'] >= len(special_ids)).sum().item() #num of labels greater than 2
        assert torch.count_nonzero(schedule_label) == valid_labels, \
            f'nonzero labels count: schedule={torch.count_nonzero(schedule_label)}, ' \
                f"valid_labels={valid_labels}, \nlabel: \n{data['label']}, '\
                    f'\nschedule_label: n\{schedule_label}"
        return scheduled_data, iters_count
      
    def add_schedule_new(self, data, schedule):
        '''
        For batchalized scheduling. PAD and EOS can be subject to unmask.
        Each sample within a batch has the same number of tokens to be unmasked.
        '''
        # take the number of tokens in the longest output sequence within a batch as the subject for scheduling.
        scheduled_data = data
        batch_size, full_len = scheduled_data['label'].size()
        schedule_label = torch.zeros(batch_size, full_len)
        # find the max output length within a batch
        max_out_len = 0
        for b in range(batch_size):
            out_len = torch.count_nonzero(data['label'][b]).item()
            if out_len > max_out_len:
                max_out_len = out_len
        unmask_num = max_out_len
        # determine the number of tokens to be unmasked for each time stamp (k)
        K = 0
        for mask_rate in schedule:
            K += 1
            u_num = max(1, int((1-mask_rate)*unmask_num))
            labeled_num = torch.count_nonzero(schedule_label[0]).item()
            sample_num = u_num - labeled_num
            # print(f'unmask_num: {unmask_num}')
            # print(f'sample_num: {sample_num}, u_num: {u_num}, labeled_num: {labeled_num}')
            for b in range(batch_size):
                unlabeled_pos = [pos for pos in range(len(schedule_label[b])) \
                    if (schedule_label[b][pos] == 0 and data['label'][b][pos] >= self.sebm.special_tok_size)]
                unmask_pos = rand.sample(unlabeled_pos, min(len(unlabeled_pos), sample_num))
                # if the tokens left are insufficient for batchalized scheduling, we randomly take the padding tokens 
                # print(f'b: {b}, schedule_label: {schedule_label}, \nlabel: {data["label"]}')
                # print(f'unlabeled_pos: {unlabeled_pos}, \nunmask_pos: {unmask_pos}')
                if len(unlabeled_pos) < sample_num:
                    # print(f'here!')
                    padding_pos = (data['label'][b] == IGNORE_INDEX).\
                        nonzero(as_tuple=True)[0].tolist()
                    # print(f'unlabeled_pos < sample_num: {unlabeled_pos} < {sample_num}, \npadding_pos: {padding_pos}')
                    unmask_pad = rand.sample(padding_pos, sample_num-len(unlabeled_pos))
                    unmask_pos.extend(unmask_pad)
                    scheduled_data['label'][b][unmask_pad] = self.sebm.tokenizer.unk_token_id #replace the PADs to be unmasked temperorily with UNK
                schedule_label[b][unmask_pos] = K
            out_mask = (data['label'] >= self.sebm.special_tok_size)
            zero_remains = (schedule_label[out_mask] == 0).sum().item()
            if zero_remains == 0:
                break
        scheduled_data['schedule_label'] = schedule_label
        return scheduled_data, K
                          
    def eval_metric(self, pred, label, metric="acc"):
        '''Calcualte the evaluation metrics for a batch pair of predictions and labels'''
        if metric == 'acc':
            matching_rows = (pred == label).all(dim=1)  # Check for equality along the rows
            return matching_rows.sum().item()  # Sum up the True values
        
    def prepare_partial_data(self, scheduled_data, order_label): #TODO: 当前只有bert版本;当前只有train版本（history tgt tokens来自sample label)
        '''
        for the diffusion-style masking process, prepare the partially masked data according to k
        return dict:
            - input: original bert_input with history tgt tokens replacing the masks
            - output: only the current tgt tokens, rest are zeros
        '''
        partial_data = {
            'input': scheduled_data['input'].clone(),
            'label': torch.zeros(scheduled_data['label'].size(), dtype=scheduled_data['label'].dtype),
            # 'segment_label': scheduled_data['segment_label'].clone().to(scheduled_data['segment_label'].dtype),
        }
        # if self.contrast:
        #     partial_data['neg_input'] = partial_data['input'].clone()
        #     partial_data['neg_label'] = partial_data['label'].clone()
        for b in range(scheduled_data['schedule_label'].size(0)):
            history_idx = ((scheduled_data['schedule_label'][b] > 0) & \
                (scheduled_data['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
            current_idx = ((scheduled_data['schedule_label'][b] > 0) & \
                (scheduled_data['schedule_label'][b] == order_label)).nonzero(as_tuple=True)[0] #<= is also workable(also fits infernce better), but results in OOM
            partial_data['input'][b][history_idx] = scheduled_data['label'][b][history_idx]
            partial_data['label'][b][current_idx] = scheduled_data['label'][b][current_idx] 
            # if self.contrast:
            #     partial_data['neg_input'][b][history_idx] = \
            #         scheduled_data['label'][b][history_idx] 
            #     partial_data['neg_label'][b][current_idx] = \
            #         scheduled_data['neg_label'][b][current_idx]
            
            
        # print(f'\nk={order_label}, xo: \n{partial_data["bert_input"][0]}, xu: \n{partial_data["bert_label"][0]}')
        return partial_data        
        
    def to_device(self, data_dict): #暂时废了
        for k,v in data_dict.items():
            if torch.is_tensor(v):
                v.to(self.device)
        return data_dict
    
    def iteration(self, epoch, data_loader, stage, train=True, schedule=None, \
                  is_ebm=True, visualize=False, parallel=False):
        '''
        Algorithm core function (exclude sampling details)
        Performs train / test / evaluation iterations during different stages
        '''
        
        avg_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        mode = "train" if train else "test"
        # initialize stat file
        eval_path = f'./ire_reasoning/stats/evaluate/{self.sebm.task_name}_' \
                    f'{self.sebm.param_type}_{self.sebm.d_model}_{stage}_stat.jsonl'
        # train_path = f'./stats/train/{self.sebm.task_name}_' \
        #             f'{self.sebm.param_type}_{self.sebm.d_model}_stat.jsonl'
        with open(eval_path, 'w') as evalfile: #, open(train_path, 'w') as trainfile:
            evalfile.write('')#, trainfile.write('')

        # progress bar
        if not parallel:
            data_iter = tqdm(
                enumerate(data_loader),
                desc="EP_%s_%s:%d" % (mode, stage, epoch),
                total=len(data_loader),
                bar_format="{l_bar}{r_bar}"
            )
        else:
            data_iter = enumerate(data_loader)
        
        if self.sebm.task_name == 'countdown':
            special_tokens = {0,1,2,3,4}
            self.sebm.special_tok_size = len(special_tokens)
        else:
            raise NotImplementedError(f'special tokens for task {self.sebm.task_name} not specified!!')
 
        # for i, (pos_data, neg_data) in data_iter: #batch
        for i, data in data_iter: #batch
            # data = self.to_device(data)
            if train:
                # Save model checkpoints every 2 iterations
                if (i % 10 == 0) and self.parallel:
                    # CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
                    if self.device == 0:
                        torch.save(self.sebm.model.state_dict(), self.ckpts_path)
                        print(f'{i}-iter ckpt saved by rank[0] to path: {self.ckpts_path}')
                        
                    # print(f'rank [{self.device}] waiting...')
                    # dist.barrier()
                    # print(f'synchronized.')
                    
                    # # configure map_location properly
                    # map_location = {'cuda:%d' % self.device: 'cuda:%d' % self.device}
                    # self.sebm.model.load_state_dict(
                    #     torch.load(self.ckpts_path, map_location=map_location, weights_only=True))
                    # print(f'rank ({self.deivce}) loaded {i}-iter ckpt')
                
                if stage == 'inference': #########for tesing
                    K = 10
                    t_list = unmasking_schedule(K+2, 'cosine')[1:-1]
                    scheduled_data, early_stop = self.add_schedule(data, t_list)
                    # print(f'\nscheduled_data: \n{scheduled_data}')
                else:
                    early_stop = 1
                ce_losses, contrast_losses, total_losses = {}, {}, {} #for storing each k-th EBM's losses for pretty wandb visualization
                for k in range(early_stop):
                    if stage == 'inference': #########for tesing
                        data = self.prepare_partial_data(scheduled_data, k+1) #AR-like
                    '''MLM training pradigm with varying t'''
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}
                    # if k == 3:
                    #     print(f'when k = 3, the data: \n{data}')
                    # neg_data = {key: value.to(self.device) for key, value in neg_data.items()}
                    if self.sebm.param_type == 'bert':
                        xo, xu = data['input'], data['label'] #already masked with rate t
                    elif self.sebm.param_type == 'gpt':
                        self.sebm.model.train()
                        # for param in self.sebm.model.parameters():
                        #     print(f'requires_grad: {param.requires_grad}')
                        xo, xu = data, data['label'] #forward argument is a dict
                    if self.contrast:
                        # neg_xo, neg_xu = data['neg_input'], data['neg_label']
                        # neg_gamma = self.sebm.forward(neg_xo, None, is_ebm=True)
                        neg_xu = data['neg_label']
                        
                    
                    gamma = self.sebm.forward(xo, None, is_ebm=True) #4,50,31 
                    
                    # # #_________________test: BERT mlm_loss___________________
                    # # print(f'\nxo.shape: {xo.shape}') #4,50
                    # mlm_output, org_loss = self.sebm.forward(xo, None, is_ebm=False) #xo is partially unmasked 'bert_input'
                    # # print(f'mlm_output.shape: {mlm_output.shape}') #Size(batch_size, |u'|, num_classes)
                    # mlm_criterion = nn.CrossEntropyLoss()
                    # # print(f'k: {k}, mlm_loss input: \n - mlm_output({mlm_output.view(-1, mlm_output.size(-1)).shape}): {mlm_output.view(-1, mlm_output.size(-1))}\n - label({xu.view(-1).shape}): {xu.view(-1)}')
                    # mlm_loss = mlm_criterion(mlm_output.view(-1, mlm_output.size(-1)), xu.view(-1))
                    # # print(f'\nmlm_loss: {mlm_loss}, loss: {org_loss}')
                    # # #———————————————————————————————————————————————————————
                    
                    # print(f'\ngamma shape: {gamma.shape}') #Size(batch_size, seq_len, num_classes)
                    
                    # _____________2-1 tentatively annotated for testing ___________
                    # 2-1. Estimate the 1D conditional logp(xu | xo) via pseudolikelihood
                    contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)
                    # recover the ignored token to padding
                    xu_ = xu.clone()
                    xu_[xu_ == -100] = 0
                    # if i == 0:
                    #     print(f'\nxu[0]: \n{xu[0]}\nxu_[0]: \n{xu_[0]}\n')
                    logp_xu_list, xu_list = [], [] #to store a batch of logp_xu's, of Size(batch_size, |u'|, num_classes)
                    neg_logp_xu_list, neg_xu_list = [], []
                    cal_t = time()
                    for r in range(xu.size(0)): #iter within batch
                        xu_token_ids = torch.nonzero(xu_[r]).squeeze() #Size(|u'|)
                        if xu_token_ids.dim()==0 and xu_token_ids:
                            xu_token_ids = torch.tensor([xu_token_ids])
                        # print(f'\n_____\nr: {r}, k: {k}\n________\n')
                        # assert torch.nonzero(xu[r]).squeeze().dim(), f"xu[r] is all zero: \n{xu[r]}"
                        if self.sebm.param_type == 'bert':
                            # logp_xu = gamma[r]#[xu_token_ids]
                            logp_xu = self.sebm.pseudolikelihood(gamma[r], xu_[r], xo[r]) #Size(|u'|, num_classes)
                            if self.contrast:
                                neg_logp_xu = self.sebm.pseudolikelihood(gamma[r], neg_xu[r], xo[r])
                        elif self.sebm.param_type == 'gpt':
                            # logp_xu = gamma[r]#[xu_token_ids]
                            logp_xu = self.sebm.pseudolikelihood(gamma[r], xu_[r], xo['input'][r]) #gamma[r]: Size(50, 31), xu[r]: Size(50), xo[r]: Size(50)
                            if self.contrast:
                                logp_xu = self.sebm.pseudolikelihood(gamma[r], neg_xu[r], xo['input'][r])
                        if (logp_xu == None):# or (logp_xu == -1): #not ebm #############################TODO: gpt需要加上后边的condition; bert要去掉后边的
                            continue #batch内该row fully unmasked, 不计算loss
                        # '''ordered, non-zero xo and xu'''
                        # label = xo[r]
                        # label[xu_token_ids] = xu[r][xu_token_ids]
                        # label_ids = label.nonzero(as_tuple=True)[0]
                        # label = label[label_ids]
                        # xu_label = label
                        # '''inordered, indexed xu'''
                        xu_label = xu[r][xu_token_ids]
                        assert logp_xu.size(0) == xu_label.size(0), \
                            f'\nlogp_xu.shape: {logp_xu.shape}, xu_label.shape: {xu_label.shape}'
                        logp_xu_list.append(logp_xu.unsqueeze(0))
                        xu_list.append(xu_label.unsqueeze(0))
                        if self.contrast:
                            neg_xu_label = neg_xu[r][xu_token_ids]
                            neg_logp_xu_list.append(neg_logp_xu.unsqueeze(0))
                            neg_xu_list.append(neg_xu_label.unsqueeze(0))
                        # print(f'logp_xu.shape: {logp_xu.shape}, xu_label.shape({xu_label.shape}): {xu_label}')
                        # if r == 0:
                        #     print(f'xu_token_ids: {xu_token_ids}, \nxu_label: {xu_label}')
                        
                        # 2-2. Contrast-loss
                        #___________sum over batch: contrast loss_______________
                        if self.contrast:
                            contrast_loss = contrast_loss + self.sebm.calculate_contrast_loss(
                                gamma[r], gamma[r], xu[r], neg_xu[r], xo[r], xo[r],
                                form="l2", threshold=1)
                        #_____________________________
                    #end of row iter
                    
                    batch_logp_xu = torch.cat(logp_xu_list, dim=1).squeeze(0) #flattened: Size(sum of |u'| within the batch, num_classes)
                    batch_xu = torch.cat(xu_list, dim=1).view(-1) #flattened: Size(sum of |u'| within the batch) 
                    assert batch_logp_xu.dim(), f'k = {k}, early_stop = {early_stop}, logp_xu_list is empty!: {logp_xu_list}, '
                    # print(f'\nbatch_logp_xu.shape:{batch_logp_xu.shape}, batch_xu.shape: {batch_xu.shape}')
                    ce_loss = self.criterion(batch_logp_xu, batch_xu)
                    contrast_loss = contrast_loss / xu.size(0) #mean over the batch
                    
                    
                    ce_is_nan = torch.isnan(ce_loss.clone().detach()).any()
                    has_zero = (batch_logp_xu > 0).any()
                    assert ce_is_nan == False, f'\nce_loss becomes NaN! '\
                        f'logp_xu: \n{batch_logp_xu[0]}, \nlogp_xu contains zero: {has_zero}\ngamma[r]: \n{gamma[r]}' \
                        f'batch_logp_xu contains nan: {torch.isnan(batch_logp_xu.clone().detach()).any()}'

                    loss = ce_loss + contrast_loss
                    # loss = mlm_loss
                    if self.train_wandb:
                        ce_losses[k] = ce_loss
                        contrast_losses[k] = contrast_loss
                        total_losses[k] = loss
                        if k == early_stop - 1: # last k (可能会因为每个sample的K不同而报错)
                            if self.parallel and self.device != 0:
                                continue
                            cal_spent = time() - cal_t
                            wandb.log({"l_ce": ce_losses, "l_contrast": contrast_losses, \
                            "l_total": total_losses, "time": cal_spent}) #loss: loss #"l_mlm": mlm_loss, \
                    # #_________mlm__________
                    if i % 50 == 0 and self.device == 0:
                        print(f'i={i}, k={k}, ce_loss: {ce_loss}, contrast_loss: {contrast_loss}, loss: {loss}')
                    # if i % 100 == 0:
                    #     pred = mlm_output.argmax(dim=-1)
                    #     correct = self.eval_metric(pred, data['bert_label'])
                    #     train_stats = {
                    #         'sample_id': i*4,
                    #         'batch_correct': correct,
                    #         'sample_loss': round(loss.item(), 2), #>10
                    #         'pred': pred.cpu().tolist()[0][self.sebm.inp_len+2:],
                    #         'label': data['bert_label'].cpu().tolist()[0][self.sebm.inp_len+2:],
                    #         'logits': [[round(ele, 2) for ele in row] for row in mlm_output.cpu().tolist()[0][self.sebm.inp_len+2:]]
                    #     }
                    #     with open(train_path, 'a') as f:
                    #         f.write(json.dumps(train_stats)+'\n')
                    # #______________________
            
                    # 3. backward and optimization only in train
                    # print(f'rank[{self.device}] start back-prop at k={k}...')
                    # print(f'rank[{self.device}] at k[{k}] waiting for back-prop...')
                    # dist.barrier()
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    # Clip gradients
                    max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(self.sebm.model.parameters(), max_norm)
                    self.optim_schedule.step_and_update_lr()
                    avg_loss += loss.cpu().item()
                    # total_correct += correct
                    # total_samples += data["is_next"].nelement()
                    post_fix = { # TODO: 不同情况k v 不同
                        "epoch": epoch,
                        "sample": i,
                        "avg_loss": avg_loss / (i + 1),
                        # "avg_acc": total_correct / total_samples * 100,
                        "loss": loss.item()
                    }
                    # print(f'rank[{self.device}] at k[{k}] finished back-prop.')
                #end of k (schedule) iter (if any)
                # if self.parallel:
                #     dist.barrier()
                #     print(f'Finished k iter, synchronized.')
                
            elif (not train) and stage == 'inference':
                '''inference with scheduled t and sequential EBMs sampling'''
                # 1. break down the sequence tokens according to the schedule
                scheduled_data, early_stop = self.add_schedule(data, schedule)
                scheduled_data = {key: value.to(self.device) for key, value in scheduled_data.items()}
                # print(f'scheduled_data: \n{scheduled_data}')
                # print(f'\nspecial_token_size: {self.sebm.special_tok_size}, vocab_size: {self.sebm.vocab_size}')
                # 2. iterate through k EBMs, send input to device, and each EBM performs gibbs sampling
                partial_pred = scheduled_data['input'].clone()
                # partial_pred = torch.randint(self.sebm.special_tok_size, \
                #         self.sebm.vocab_size, data['bert_label'].size()) #init
                # partial_pred[:, :(self.sebm.inp_len+2)] = 0
                # partial_pred[:, -1] = 0
                
                # invalid_pos = data['bert_label'] < len(special_tokens)
                # print(f'invalid_pos: \n{invalid_pos}')
                # partial_pred[invalid_pos] = data['bert_label'][invalid_pos]
                partial_pred[partial_pred == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
                # print(f'early_stop: {early_stop}, initial partial_pred({partial_pred.shape}): \n{partial_pred}')
                # partial_pred = torch.zeros_like(scheduled_data['bert_input']) #init
                k_losses, k_energies = {}, {}
                for k, t in enumerate(schedule):
                    
                    # if k != len(schedule)-1:
                    #     continue
                    
                    # print(f'\n_________\nk = {k}:\n')
                    partial_pred, sth = self.sebm.sampling_new( #sth: loss_list
                        k+1, #1-indexed unmasking order label
                        partial_pred,
                        scheduled_data, 
                        self.sampler,
                        self.sampling_times,
                        visualize=visualize,
                        # groundtruth=scheduled_data['label'][0], #the first sample's label for visualizing the landscape
                        batch_id=i,
                    )
                    # print(f'{k}-th partial_pred: \n{partial_pred},\nloss_list: {sth}')
                    # if not visualize:
                    #     k_losses[str(k)], k_energies[str(k)] = sth['losses'], sth['energies']
                    if k+1 == early_stop: #fully unmasked before reaching k
                        break
                pred = partial_pred
                invalid_pos = scheduled_data['label'] < len(special_tokens)
                pred[invalid_pos] = scheduled_data['label'][invalid_pos]
                correct = self.eval_metric(pred, scheduled_data['label'])
                total_correct += correct
                total_samples = (i+1)*scheduled_data['input'].size(0)
                # print(f"pred: \n{pred}, \nlabel: \n{scheduled_data['bert_label']},\ncorrrect: {correct}")
                # ______decode______
                label = scheduled_data['label'][0].clone()
                label[label == -100] = 0
                pred[0][pred[0] == -100] = 0
                decoded_label = self.sebm.tokenizer.decode(label.tolist(), skip_special_tokens=True)
                decoded_pred = self.sebm.tokenizer.decode(pred[0].tolist(), skip_special_tokens=True)
                if i < 10:
                    print(f'i:{i}, \ndecoded label: \n{decoded_label}, \ndecoded pred: \n{decoded_pred},\ncorrrect: {correct}\n\n')
                # __________________
                # TODO 单独记录energy变化
                flattened_loss, flattened_energy = [], []
                for k in k_losses:
                    flattened_loss.extend(k_losses[k])
                    flattened_energy.extend(k_energies[k])
                post_fix = {
                    "sample": i,
                    "acc": round(total_correct*100/total_samples, 2),
                    "avg_losses": flattened_loss,
                    "avg_energies": flattened_energy,
                }
                if self.test_wandb:
                    wandb.log({'acc': "acc"})
                with open(eval_path, 'a') as statsfile:
                    statsfile.write(json.dumps(post_fix)+'\n')
            
            elif (not train) and stage == 'sft':
                data['input'], data['label'] = data['input'].to(self.device), data['label'].to(self.device)
                # print(f'data: \n{data}')
                if self.sebm.param_type == 'bert':
                    logits, _ = self.sebm.forward(data['input'], None, is_ebm=False) # data['segment_label']
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data['label'].view(-1)) #flattened (batch_size * seq_len)
                    # print(f'logits({logits.shape}): \n{logits}')
                    pred = logits.argmax(dim=-1) #argmin for "_mlm_test.pth"(训反了, energy漏加-); argmax for "_w_mask.pth"?
                    invalid_pos = data['label'] < len(special_tokens)
                    pred[invalid_pos] = data['label'][invalid_pos]
                    # print(f"logits.shape: {logits.shape}, label.shape: {data['bert_label'].shape}")
                    correct = self.eval_metric(pred, data['label'])
                    # print(f"pred: \n{pred}, \nlabel: \n{data['bert_label']},\ncorrrect: {correct}")
                    # # ______decode______ 
                    # label = data['bert_label'][0].clone()
                    # label[label == -100] = 0
                    # pred[0][pred[0] == -100] = 0
                    # decoded_label = self.sebm.tokenizer.decode(label.tolist(), skip_special_tokens=True)
                    # decoded_pred = self.sebm.tokenizer.decode(pred[0].tolist(), skip_special_tokens=True)
                    # print(f'i:{i}, \ndecoded label: \n{decoded_label}, \ndecoded pred: \n{decoded_pred},\ncorrrect: {correct}\n\n')
                    # # __________________
                elif self.sebm.param_type == 'gpt':
                    logits, org_loss = self.sebm.forward(data)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data['labels'].view(-1))
                    pred = logits.argmax(dim=-1)
                    invalid_pos = data['labels'] < len(special_tokens)
                    pred[invalid_pos] = data['labels'][invalid_pos]
                    # print(f"logits.shape: {logits.shape}, label.shape: {data['labels'].shape}")
                    correct = self.eval_metric(pred, data['labels'])
                    # print(f"pred: \n{pred}, \nlabel: \n{data['labels']},\ncorrrect: {correct}")
                    # ______decode______ 
                    label = data['labels'][0].clone()
                    label[label == -100] = 0
                    pred[0][pred[0] == -100] = 0
                    decoded_label = self.sebm.tokenizer.decode(label.tolist(), skip_special_tokens=True)
                    decoded_pred = self.sebm.tokenizer.decode(pred[0].tolist(), skip_special_tokens=True)
                    print(f'i: {i}, decoded label: \n{decoded_label}, \ndecoded pred: \n{decoded_pred}')
                    # __________________
                else:
                    raise NotImplementedError
                avg_loss += loss.item()
                total_correct += correct
                if self.sebm.param_type == 'bert':
                    total_samples += data['input'].size(0)
                elif self.sebm.param_type == 'gpt':
                    total_samples += data['input_ids'].size(0)
                if self.test_wandb:
                    wandb.log({"l_ce": loss, 'org_loss': org_loss, "avg_acc": total_correct/total_samples * 100})
                
                # save stats to jsonl file
                stats = {
                    'sample_id': i*4,
                    'batch_correct': correct,
                    'sample_loss': round(loss.item(), 2), #>10
                    # 'decoded_label': decoded_label,
                    # 'decoded_pred': decoded_pred,
                    'pred': pred.cpu().tolist()[0],
                    'logits': [[round(ele, 2) for ele in row] for row in logits.cpu().tolist()[0]]
                }
                if self.sebm.param_type == 'bert':
                    stats['label'] = data['label'].cpu().tolist()[0]
                elif self.sebm.param_type == 'gpt':
                    stats['label'] = data['labels'].cpu().tolist()[0]
                with open(eval_path, 'a') as statsfile:
                    statsfile.write(json.dumps(stats)+'\n')

            else:
                '''test the MLM training effectiveness and sampling'''
                # TODO 或许没用？
                raise NotImplementedError
                
            # break ###############test
            # if i == 10:
            if i >= 2:
                break ###############test
        #end of batch iter


        if train:
            print(f'\n\nFinished training.')
            # ckpts_path = f'./ire_reasoning/ebm_ckpts/{self.sebm.task_name}_{self.sebm.param_type}{self.sebm.d_model}_{stage}_ebm_test.pth' #w_mask_inverse_test
            if not self.parallel:
                torch.save(self.sebm.model.state_dict(), self.ckpts_path)
                print(f'\nmodel saved to {self.ckpts_path}')
            else: #if parallel, save the checkpoints from one of the processes
                print(f'\nrank({self.device}) finished training! waiting...')
                dist.barrier()
                print(f'synchronized...')
                if self.device == 0:
                    torch.save(self.sebm.model.state_dict(), self.ckpts_path)
                    print(f'\nrank[0] model finally saved ckpt to {self.ckpts_path}')
        elif (not train) and (schedule is not None):
            final_acc = round(total_correct*100/total_samples, 2)
            print(f'\nFinished sampling (inference), '\
                f'final accuracy: {final_acc}')
            with open(eval_path, 'a') as statsfile:
                statsfile.write(f'\nFinal Accuracy: {final_acc}\n')
            print(f'\nStats written to path: {eval_path}')
                
    def fast_iteration(self, epoch, data_loader, stage, train=True, visualize=False, parallel=False):
        '''
        Faster training featured with new sequential EBM and batchalized pseudolikelihood function
        '''   
        if train:
            mode = 'train'
        else:
            mode = 'inference'
        if not parallel or (parallel and self.device==0):
            data_iter = data_iter = tqdm(
                enumerate(data_loader),
                desc="EP_%s_%s:%d" % (mode, stage, epoch),
                total=len(data_loader),
                bar_format="{l_bar}{r_bar}"
            )
        else:
            data_iter = enumerate(data_loader)
        if self.sebm.task_name == 'countdown':
            special_tokens = {0, 1, 2, 3, 4}
            self.sebm.special_tok_size = len(special_tokens)
        else:
            raise NotImplementedError
        if (not train) and stage == 'inference':
            total_correct, total_samples = 0, self.test_size
            # initialize stat file
            eval_path = f'./ire_reasoning/stats/evaluate/{self.sebm.task_name}_' \
                        f'{self.sebm.param_type}_{self.sebm.d_model}_{stage}_stat.jsonl'
            with open(eval_path, 'w') as evalfile: 
                evalfile.write('')
        
        # if self.device == 0: ####test
        #     print(f'before iter on batches, {torch.cuda.memory_summary()}')
        for i, data in data_iter:
            # prepare unmasking schedule
            max_K = 10
            t_list = unmasking_schedule(max_K+2, 'cosine')[1:-1]
            scheduled_data, K = self.add_schedule_new(data, t_list)
            # print(f"schedule_label: \n{scheduled_data['schedule_label']}")
            # raise
            if (not train) and stage == 'inference':
                partial_pred = scheduled_data['input'].clone()
                partial_pred[partial_pred == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
                
            # iter through the K EBMs
            ce_losses, contrast_losses, total_losses = {}, {}, {}
            for k in range(K):
                # if i < 10:
                #     print(f'\n___________k={k}___________\n')
                data = self.prepare_partial_data(scheduled_data, k+1) #AR-like, the previously unmasked tokens won't be predicted again
                data = {k:v.to(self.device) for k,v in data.items()}
                u = self.sebm.get_2D_indices( #just to remove the zero paddings
                    array=data['label'],
                    val=self.sebm.tokenizer.pad_token_id,
                    type='remove_pad',
                ) #Size(batch_size, |u|)
                if train:
                    # for parallel training, save the checkpoints per 10 iterations
                    if self.parallel and (i % 10 == 0) and (k == 0) and self.device == 0:
                        # CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
                        # print(f'\nbefore saving, {torch.cuda.memory_summary()}')
                        # print(f'cuda{self.device} waiting for sync...')
                        # dist.barrier()
                        torch.save(self.sebm.model.state_dict(), self.ckpts_path)
                        # print(f'\nafter saving, {torch.cuda.memory_summary()}')
                    
                    torch.autograd.set_detect_anomaly(True)
                    assert self.contrast, f'Fast iteration requires enabling contrast loss!'
                    xo, xu = data['input'], data['label']
                    xu[xu == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id #reset the IGNORE tokens back to PAD
                    # print(f'\nxo({xo.shape}): {xo}\nxu({xu.shape}): {xu}')
                    cal_t = time()
                    # calculate losses
                    logp_xu, energy_dist = self.sebm.pseudolikelihood_revised(xo, xu) #both of (B,V,U)
                    ce_loss = self.criterion(logp_xu, xu.gather(dim=1, index=u))
                    # if i < 10:
                    #     print(f'ce_loss({ce_loss.shape}): {ce_loss}')
                    if self.contrast:
                        #_____contrast loss hyperparams______
                        beta = 1
                        threshold = 2
                        #____________________________________
                        contrast_loss = self.sebm.contrast_loss_revised(energy_dist, xu, threshold=threshold, mode='hinge')
                        # contrast_loss = self.sebm.fast_contrast_loss(xo, xu, loss_type='l2', threshold=2)
                        # if i < 10:
                        #     print(f'contrast_loss({contrast_loss.shape}): {contrast_loss}')
                        loss = ce_loss + beta*contrast_loss
                    else:
                        loss = ce_loss
                    
                    # logging
                    if self.train_wandb:
                        ce_losses[k], total_losses[k] = ce_loss, loss
                        if self.contrast:
                            contrast_losses[k] = contrast_loss
                        if k == K-1: #last k
                            if self.parallel and self.device != 0:
                                continue
                            cal_spent = time() - cal_t
                            wandb.log({'ce_loss': ce_losses, 'contrast_loss': contrast_losses, \
                                'loss':total_losses, 'time': cal_spent})
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    # clip gradients
                    max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(self.sebm.model.parameters(), max_norm)
                    self.optim_schedule.step_and_update_lr()
                elif (not train) and stage == 'inference':
                    self.sebm.model.eval()
                    with torch.no_grad():
                        # if i < 10:
                        #     print(f'\n___________inference with k={k}-th EBM___________\n')
                        partial_pred, sth = self.sebm.sampling_revised(
                            partial_pred,
                            u,
                            self.sampler,
                            self.sampling_times,
                            visualize=visualize,
                            batch_id=k, ##########this should be i, currently use k for testing 
                        )
                    # end of eval mode
            # end K (EBM) iter
            if (not train) and stage == 'inference': # decode and logging
                pred = partial_pred
                pred, scheduled_data['label'] = pred.to(self.device), scheduled_data['label'].to(self.device)
                invalid_pos = scheduled_data['label'] < len(special_tokens)
                pred[invalid_pos] = scheduled_data['label'][invalid_pos]
                correct = self.eval_metric(pred, scheduled_data['label']) #TODO: add bachalization
                total_correct += correct
                label = scheduled_data['label'][0].clone()
                label[label == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
                pred[0][pred[0] == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
                decoded_label = self.sebm.tokenizer.decode(label.tolist(), skip_special_tokens=True)
                decoded_pred = self.sebm.tokenizer.decode(pred[0].tolist(), skip_special_tokens=True)
                if i < 10:
                    print(f'i: {i}, \ndecoded label: {decoded_label}, \ndecoded pred: {decoded_pred}, \ncorrect: {correct}')
                stat = {
                    'batch_id': i,
                    'correct': correct,
                    'label': decoded_label,
                    'pred': decoded_pred
                }
                with open(eval_path, 'a') as statfile:
                    statfile.write(json.dumps(stat)+'\n')
            # break #################test
            # if train and (i > 4): ##################
            #     break ##################
        # end of batch iter
        # if train: #save checkpoints
        #     if not self.parallel:
        #         print(f'\n\nFinished training!')
        #         torch.save(self.sebm.model.state_dict(), self.ckpts_path)
        #         print(f'\nmodel saved to {self.ckpts_path}.')
        #     elif (epoch == self.epochs-1) and self.parallel: #TODO: parallel issue: somehow cannot synchronize...
        #         print(f'\nThis is last epoch({epoch}), rank({self.device}) finished training! waiting...')
        #         dist.barrier()
        #         print(f'synchronized...')
        #         if self.device == 0:
        #             torch.save(self.sebm.model.state_dict(), self.ckpts_path)
        #             print(f'\nrank[0] model finally saved ckpt to {self.ckpts_path}')
        #     else: #parallel, non-last epochs
        #         print(f'\nEpoch {epoch} completed on rank{self.device}, go to next epoch.\n')
        if (not train): #show inference performance
            final_acc = round(total_correct*100/total_samples, 2)
            print(f'\n\nFinished sampling!'\
                f'\nFinal accuracy: {final_acc}, total_correct/total_samples = {total_correct}/{total_samples}')
            with open(eval_path, 'a') as statfile:
                statfile.write(f'\nFinal Accuracy: {final_acc}\n')
            print(f'\nStats written to path: {eval_path}')
                
                
                
                
                

@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config):
    print(f'\n___\nStage: {config.train.stage}\n___\n')
    '''1. Load task datasets'''
    if config.task_name.startswith('binary'):
        max_len = config.tasks[config.task_name].inp_len + config.tasks[config.task_name].out_len #+ 3
    elif config.task_name == 'countdown':
        max_len = config.tasks[config.task_name].max_len
    train_loader, test_loader, train_size, test_size, tokenizer = load_data(
        config.task_name, 
        stage='inference',#config.train.stage, #这里声明了stage: pretrain / sft
        max_len=max_len,
        train_batch_size=config.train.batch_size, 
        val_batch_size=config.sampling.batch_size,
        contrast=config.train.contrast,
        parallel=config.parallel,
    )
    print(f'param type: {config.param_type}')
    print(f'\nLoaded datasets for task: {config.task_name}, max_len: {max_len}, ' \
        f'train:test={train_size}:{test_size},'\
        f'batch_size={config.train.batch_size}:{config.sampling.batch_size}...')
    # return ##############

    '''2. Initialize and train EBMs'''
    print(f'\nInitializing EBMs...')
    task_config = config.tasks[config.task_name]
    if task_config.name.startswith('binary'):
        print(f'inp_len: {task_config.inp_len}, out_len: {task_config.out_len}, num_classes: {task_config.num_classes}')
    if config.param_type == 'bert':
        d_model = config.models[config.param_type].d_model
        n_layers = config.models[config.param_type].n_layers
        heads = config.models[config.param_type].heads
        sebm = BERTSequentialEBMs( #fixed input and output lengths
            tokenizer=tokenizer,
            task_config=task_config,
            d_model=d_model, #######32 hidden_size
            n_layers=n_layers,
            heads=heads,
            device=config.device
            )
        # assert all(param.device.type == config.device for param in sebm.model.parameters()), \
        #     f'All param devices: \n{[param.device.type for param in sebm.model.parameters()]}'
    elif config.param_type == 'gpt': #gpt2-6m-scratch from diffu-vs-ar paper
        sebm = GPTSequentialEBMs( #flexible input and output length
            tokenizer,
            task_config,
            model_config=None, #TODO: other configs, if necessary
            device=config.device, #TODO
        )
    elif config.param_type == 'fast': #largely the same hyper params as bert sebm
        d_model = config.models['bert'].d_model
        n_layers = config.models['bert'].n_layers
        heads = config.models['bert'].heads
        sebm = FastSequentialEBMs(
            tokenizer=tokenizer,
            task_config=task_config,
            d_model=d_model,
            n_layers=n_layers,
            heads=heads,
            device=config.device,
        )
    else:
        raise NotImplementedError
    sebm_trainer = SequentialEBMsTrainer(
        sebm,
        train_loader,
        test_loader,
        config.train.lr,
        train_size=train_size,
        test_size=test_size,
        train_wandb=config.train.wandb,
        sampler=config.sampling.sampler,
        test_wandb=config.sampling.wandb,
        is_ebm=config.is_ebm,
        contrast=config.train.contrast,
        sampling_times=config.sampling.times,
        device=config.device, #可能是这里导致报错
        parallel=config.parallel,
    )

    # return ##############
    '''3. Train & Inference'''
    if not config.load_ebm_ckpts:
        print(f'No checkpoints found, start training from scratch.')
        # print(f'\nBefore training...')
        # # test(sebm, val_data[0]) 
        # k=10
        # sebm_trainer.evaluate(k, stage=config.sampling.stage, visualize=False)
        print(f'\n\n\n3. Start training...')
        if config.train.wandb:
            wandb.login()
            run = wandb.init(
                project=f'EBM_train-{task_config.name}_{config.param_type}',  # Specify your project
                config={                        # Track hyperparameters and metadata
                    "learning_rate": config.train.lr,
                    "epochs": config.train.epochs,
                },
            )
        '''3.0 Set the early_stopper'''
        early_stopper = EarlyStopper(patience=25, min_delta=5e-4, ema_beta=0.9, mode='min')
        for epoch in range(config.train.epochs):
            '''3.1 train one epoch'''
            # sebm_trainer.train_data.sampler.set_epoch(epoch) #make shuffling work properly across multiple epochs
            sebm_trainer.train(epoch, config.train.stage)
            # if sebm_trainer.train_is_converged: #TODO: remove
            #     print(f'\nend epochs loop')
            #     break
            
            '''3.2 validate converge or not'''
            if sebm_trainer.device == 0:
                print(f'\n\nEpoch{epoch} finished, start validate!\n\n')
            converge, val_acc, val_ce = sebm_trainer.validate(epoch, early_stopper)
            if converge or (epoch == sebm_trainer.epochs-1):
                if sebm_trainer.device == 0:
                    print(f'\n\n你converged!!\nepoch: {epoch}\nval_acc: {val_acc}\nval_ce: {val_ce}\n\n')
                '''3.3 save checkpoints'''
                if not config.parallel:
                    print(f'\n\nFinished training!')
                    torch.save(sebm_trainer.sebm.model.state_dict(), sebm_trainer.ckpts_path)
                    print(f'\nmodel saved to {sebm_trainer.ckpts_path}.')
                # else: #TODO: parallel issue: somehow cannot synchronize...
                #     print(f'\nThis is last epoch({epoch}), rank({sebm_trainer.device}) finished training! waiting...')
                #     dist.barrier()
                #     print(f'synchronized...')
                #     if sebm_trainer.device == 0:
                #         torch.save(sebm_trainer.sebm.model.state_dict(), sebm_trainer.ckpts_path)
                #         print(f'\nrank[0] model finally saved ckpt to {sebm_trainer.ckpts_path}')
                break
            else:
                print(f'\n\nNot convreged...\nepoch: {epoch}\nval_acc: {val_acc}\nval_ce: {val_ce}\n\n')
        # return##############
    else:
        ckpts_path = sebm_trainer.ckpts_path
        # ckpts_path = f'./ire_reasoning/ebm_ckpts/{task_config.name}_{config.param_type}{config.models[config.param_type].d_model}_{config.train.stage}_ebm_test.pth'
        sebm_trainer.load_model(ckpts_path, device=config.device)##########################
        print(f'\n3. Checkpoints loaded from {ckpts_path}...')
        
        if config.continue_train: #further tune the mlm model with fully masked t2 ('sft' stage)
            print(f'Continue train on {config.train.stage}...')
            if config.train.wandb:
                wandb.login()
                run = wandb.init(
                    project=f'EBM_continue_train-{task_config.name}_{config.param_type}',  # Specify your project
                    config={                        # Track hyperparameters and metadata
                        "learning_rate": config.train.lr,
                        "epochs": config.train.epochs,
                    },
                )
            for epoch in range(config.train.epochs):
                sebm_trainer.train(epoch, config.train.stage)
            
        

    # return ###############

    '''3. Evaluate'''
    k = 10
    if config.sampling.wandb:
        wandb.login()
        run = wandb.init(
            project=f'EBM_eval-{task_config.name}_{config.param_type}',  # Specify your project
            config={                        # Track hyperparameters and metadata
                "k": k,
                "batch_size": config.sampling.batch_size,
            },
        )
    print(f'\n\n\n4. Start evaluation...')
    if config.sampling.stage == 'inference':
        print(f'sampler: "{config.sampling.stage}"')
    sebm_trainer.evaluate(k, stage=config.sampling.stage, visualize=config.visualize)


# Example usage
if __name__ == "__main__":
    main()