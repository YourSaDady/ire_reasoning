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
import transformers
from transformers import AutoConfig
# print(f'The current working directory: {os.getcwd()}')
import hydra
from sequential_ebms import BERTSequentialEBMs, GPTSequentialEBMs, FastSequentialEBMs
from dataset import load_data
from utils import convert_time, VisualizeEBMs
import random as rand
from time import time
import json
import math
import threading, copy

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
        # lr = self.init_lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
    # def fast_forward(self, n_steps: int):
    #     '''adjust the lr to the n-th step in one-run'''
    #     self.n_current_steps = n_steps
    #     lr = self.init_lr * self._get_lr_scale()
    #     for param_group in self._optimizer.param_groups:
    #         param_group['lr'] = lr

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
        log_train=True,
        log_test=False,
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
        self.log_train = log_train
        self.log_test = log_test
        self.epochs = epochs
        print(f"Total Parameters: {sum([p.nelement() for p in self.sebm.model.parameters()])}, is_ebm: {self.is_ebm}")
        self.ckpts_path = f'./ebm_ckpts/{self.sebm.task_name}_{self.sebm.param_type}_{self.sebm.model_arc}_{self.sebm.model_scale}_10k.pth' # _ebm8.9 (sota with sampling_new())
        self.local_log_prefix = f'./matplotlib/{self.sebm.task_name}_{self.sebm.param_type}_{self.sebm.model_arc}_{self.sebm.model_scale}_10k' #replace wandb, visualize the log via matplotlib
        
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
            special_tokens = {0, 1, 2, 3, 4}
            self.sebm.special_tok_size = len(special_tokens)
            
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
                
                # break#############################
            # end of batch iter
            val_ce = val_loss / total_samples
            final_acc = round(total_correct*100/total_samples, 2)
            
            # gather the global validation accuracy
            # total_correct = torch.tensor(total_correct, dtype=torch.long, device=self.device)
            # total_samples = torch.tensor(total_samples, dtype=torch.long, device=self.device)
            # dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            # dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # final_acc = (total_correct.float() / total_samples).item()
            if self.parallel and dist.get_rank() == 0:
                print(f'\n\nInside validate(), local val_acc: {final_acc:.2f}')
                if final_acc >= 98.2:
                    torch.save(self.sebm.model.state_dict(), self.ckpts_path)
                    print(f'acc: {final_acc} is over 98%, saved to {self.ckpts_path}\n\n')
            
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
                    try:
                        unmask_pad = rand.sample(padding_pos, sample_num-len(unlabeled_pos))
                    except:
                        print(f'unlabeled_pos < sample_num: {unlabeled_pos} < {sample_num}, \npadding_pos: {padding_pos}')
                        raise
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
        for b in range(scheduled_data['schedule_label'].size(0)):
            history_idx = ((scheduled_data['schedule_label'][b] > 0) & \
                (scheduled_data['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
            current_idx = ((scheduled_data['schedule_label'][b] > 0) & \
                (scheduled_data['schedule_label'][b] == order_label)).nonzero(as_tuple=True)[0] #<= is also workable(also fits infernce better), but results in OOM
            partial_data['input'][b][history_idx] = scheduled_data['label'][b][history_idx]
            partial_data['label'][b][current_idx] = scheduled_data['label'][b][current_idx] 
            
            
        # print(f'\nk={order_label}, xo: \n{partial_data["bert_input"][0]}, xu: \n{partial_data["bert_label"][0]}')
        return partial_data        
        
    def to_device(self, data_dict): #暂时废了
        for k,v in data_dict.items():
            if torch.is_tensor(v):
                v.to(self.device)
        return data_dict
    
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
            
        # data_iter = tqdm(
        #     enumerate(data_loader), 
        #     desc="EP_%s_%s:%d" % (mode, stage, epoch),
        #     total=len(data_loader),
        #     bar_format="{l_bar}{r_bar}",
        #     disable=(self.device!=0), 
        #     mininterval=2
        # )
        
        special_tokens = {0, 1, 2, 3, 4}
        self.sebm.special_tok_size = len(special_tokens)
        if (not train) and stage == 'inference':
            total_correct, total_samples = 0, self.test_size
            # initialize stat file
            eval_path = f'./stats/evaluate/{self.sebm.task_name}_' \
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
            # print(f"scheduled_data: \ninput: {scheduled_data['input'][0]}\nlabel: {scheduled_data['label'][0]}\nschedule_label: {scheduled_data['schedule_label'][0]}")
            if (not train) and stage == 'inference':
                partial_pred = scheduled_data['input'].clone()
                partial_pred[partial_pred == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
                
            # iter through the K EBMs
            ce_losses, contrast_losses, total_losses = {}, {}, {}
            for k in range(K):
                # if i < 10:
                #     print(f'___________i={i}, k={k}___________')
                data = self.prepare_partial_data(scheduled_data, k+1) #AR-like, the previously unmasked tokens won't be predicted again
                data = {k:v.to(self.device) for k,v in data.items()}
                try:
                    u = self.sebm.get_2D_indices( #just to remove the zero paddings
                        array=data['label'],
                        val=self.sebm.tokenizer.pad_token_id,
                        type='remove_pad',
                    ) #Size(batch_size, |u|)
                except:
                    print(f"data['label']: {data['label']}")
                    raise
                if train:
                    # for parallel training, save the checkpoints per 10 iterations
                    if self.parallel and (i % 10 == 0) and (k == 0) and self.device == 0:
                        # CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
                        # print(f'\nbefore saving, {torch.cuda.memory_summary()}')
                        # print(f'cuda{self.device} waiting for sync...')
                        # dist.barrier()
                        tmp_state = copy.deepcopy(self.sebm.model.state_dict())
                        # torch.save(self.sebm.model.state_dict(), self.ckpts_path)
                        threading.Thread(target=lambda: torch.save(tmp_state, self.ckpts_path)).start()
                        # print(f'\nckpts saved')
                        # print(f'\nafter saving, {torch.cuda.memory_summary()}')
                    
                    # torch.autograd.set_detect_anomaly(True) #这里取消注释会卡住。。。
                    # assert self.contrast, f'Fast iteration requires enabling contrast loss!'
                    xo, xu = data['input'], data['label']
                    
                    # xu[xu == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id #reset the IGNORE tokens back to PAD
                    
                    # print(f'\nxo({xo.shape}): {xo}\nxu({xu.shape}): {xu}')
                    # print(f'u({u.shape}): {u}')
                    # raise
                    cal_t = time()
                    # calculate losses
                    logp_xu, energy_dist = self.sebm.pseudolikelihood_revised(xo, xu) #both of (B,V,U), paddings are manually assigned high energy
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
                    ce_losses[k], total_losses[k] = ce_loss, loss
                    if self.contrast:
                        contrast_losses[k] = contrast_loss
                    if k == K-1: # last k
                        if self.log_train: # and self.device == 0 ##############################
                            if self.parallel and self.device != 0:
                                continue
                            elif self.parallel==False or self.device == 0:
                                cal_spent = time() - cal_t
                                log = {
                                    'i': i,
                                    'cal_spent': cal_spent,
                                    'total_losses': {k: v.detach().cpu().item() for k, v in total_losses.items()},
                                    'ce_losses': {k: v.detach().cpu().item() for k, v in ce_losses.items()},
                                }
                                if self.contrast:
                                    log['contrast_losses'] = {k: v.detach().cpu().item() for k, v in contrast_losses.items()}
                                with open(f'{self.local_log_prefix}_loss.jsonl', 'a', encoding='utf-8') as logfile:
                                    logfile.write(json.dumps(log) + '\n')
                        if (i % 20 == 0):
                            avg_ce_loss = torch.stack(list(ce_losses.values())).mean()
                            if self.contrast:
                                avg_contrast_loss = torch.stack(list(contrast_losses.values())).mean()
                            else: 
                                avg_contrast_loss = -1
                            avg_loss = torch.stack(list(total_losses.values())).mean()
                            print(f'avg_loss: {avg_loss}, avg_ce_loss: {avg_ce_loss}, avg_contrast_loss: {avg_contrast_loss}')
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    # clip gradients
                    max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(self.sebm.model.parameters(), max_norm)
                    self.optim_schedule.step_and_update_lr()
                    # dist.barrier()
                    # print(f'rank{self.device} synchronized after optimizer.step()')
                elif (not train) and stage == 'inference':
                    self.sebm.model.eval()
                    with torch.no_grad():
                        # if i < 10:
                        #     print(f'\n___________inference with k={k}-th EBM___________\n')
                        partial_pred[partial_pred == IGNORE_INDEX] = self.sebm.tokenizer.pad_token_id
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
            # print(f'\niter i={i} finished')
            # # raise
            # if train and (i > 10): ##################
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
        print(f'device{self.device} finished!')
        # raise ###########################################
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
    # Set the project root
    os.chdir(config.project_root)
    
    print(f'\n___\nStage: {config.train.stage}\n___\n')
    '''1. Load task datasets'''
    if config.task_name.startswith('binary'):
        max_len = config.tasks[config.task_name].inp_len + config.tasks[config.task_name].out_len #+ 3
    elif config.task_name.startswith('cd') or config.task_name == 'sudoku':
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
    print(f'\nparam type: {config.param_type}, model_arc: {config.model_arc}, model_scale: {config.model_scale}')
    print(f'\nLoaded datasets for task: {config.task_name}, max_len: {max_len}, ' \
        f'train:test={train_size}:{test_size},'\
        f'batch_size={config.train.batch_size}:{config.sampling.batch_size}...')
    # return ##############

    '''2. Initialize and train EBMs'''
    print(f'\nInitializing EBMs...')
    task_config = config.tasks[config.task_name]
    if task_config.name.startswith('binary'): #TODO: delete
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
        # d_model = config.models['tiny'].d_model
        # n_layers = config.models['tiny'].n_layers
        # heads = config.models['tiny'].heads
        model_config_path = f'./models/model_config_{config.model_scale}'
        model_config = AutoConfig.from_pretrained(model_config_path)
        sebm = FastSequentialEBMs( #_build_model can choose BERT-from-scratch, or naive-GPT2-miny
            tokenizer=tokenizer,
            task_config=task_config,
            model_config=model_config,
            model_arc=config.model_arc,
            model_scale=config.model_scale,
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
        log_train=config.train.local_log,
        log_test=config.sampling.local_log,
        sampler=config.sampling.sampler,
        is_ebm=config.is_ebm,
        contrast=config.train.contrast,
        sampling_times=config.sampling.times,
        device=config.device, #可能是这里导致报错
        parallel=config.parallel,
        warmup_steps=config.tasks[config.task_name].warmup_steps,
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
            for epoch in range(config.train.epochs):
                sebm_trainer.train(epoch, config.train.stage)
            
        

    # return ###############

    '''3. Evaluate'''
    k = 10
    print(f'\n\n\n4. Start evaluation...')
    if config.sampling.stage == 'inference':
        print(f'sampler: "{config.sampling.stage}"')
    sebm_trainer.evaluate(k, stage=config.sampling.stage, visualize=config.visualize)


# Example usage
if __name__ == "__main__":
    main()