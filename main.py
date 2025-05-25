'''
Train and Evaluate sequential EBMs
'''

import torch
import torch.nn as nn
import torch.optim as optim
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
from sequential_ebms import BERTSequentialEBMs, GPTSequentialEBMs
from dataset import load_bert_data, load_data, load_gpt_data
from utils import convert_time, VisualizeEBMs
from transformers import BertTokenizer, AutoConfig, AutoModelForCausalLM
import random as rand
import wandb
import json

IGNORE_INDEX = -100

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
        train_wandb=True,
        test_wandb=False,
        is_ebm=True,
        device='cpu' ########debugging
    ):
        self.device = device
        self.sebm = sebm
        self.is_ebm = is_ebm
        self.train_data = train_dataloader
        self.test_data = test_dataloader
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
        print(f"Total Parameters: {sum([p.nelement() for p in self.sebm.model.parameters()])}, is_ebm: {self.is_ebm}")
    
    def load_model(self, ckpts_path):
        state_dict = torch.load(ckpts_path)
        self.sebm.model.load_state_dict(state_dict)
        
    def train(self, epoch, stage):
        self.iteration(epoch, self.train_data, stage, is_ebm=self.is_ebm)

    def test(self, epoch, stage): #暂时无用
        self.iteration(epoch, self.test_data, stage, train=False)
        
    def evaluate(self, k, scheduler='cosine', stage='pretrain', visualize=False): #Inference
        '''
        Recover a fully masked sequence using a specified scheduler, with decreasing t
        '''
        t_list = unmasking_schedule(k+2, scheduler)[1:-1]
        print(f't_list: {t_list}')
        if visualize:
            raise NotImplementedError
        else:
            self.iteration(1, self.test_data, stage=stage, train=False, \
                schedule=t_list)
        
    def add_schedule(self, data, schedule):
        '''
        simply add a schedule label key-value pair to the data dict
        
        params:
            data: batchalized data dict with bert_inputs, bert_labels, segment_labels and is_positive
            schedule: a list of decreasing masking rate t (fully masked to fully unmasked)
        return:
            scheduled_data: batchalized data dict extended with schedule_label k-v pair
        '''
        scheduled_data = data
        batch_size, io_len = data['bert_label'].size()
        # print(f"bert_label({data['bert_label'].shape}): {data['bert_label']}")
        schedule_label = [[0 for c in range(io_len)] for r in range(batch_size)] #initialize a 2D batchalized schedule label (4,30)
        special_ids = {0,1,2,3,4} #custom_tokenizer: pad, sep, mask, eos, unk
        # special_ids = {0} #mask
        full_label = data['bert_label'].tolist()
        assert len(full_label)==batch_size and len(full_label[0])==io_len, \
            f'bert_label: Size({len(full_label)}, {len(full_label[0])})'
        iters_count = 0 #early stop, can be lesser than k
        for k, t in enumerate(schedule):
            iters_count += 1
            unmask_num = int((1-t) * torch.count_nonzero(data['bert_label'][0]))
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
        scheduled_data['schedule_label'] = schedule_label
        # 以下assertion变了： torch.count_nonzero(data['bert_label'])
        valid_labels = (data['bert_label'] >= len(special_ids)).sum().item() #num of labels greater than 2
        assert torch.count_nonzero(schedule_label) == valid_labels, \
            f'nonzero labels count: schedule={torch.count_nonzero(schedule_label)}, ' \
                f"bert={valid_labels}, \nbert_label: \n{data['bert_label']}, '\
                    f'\nschedule_label: n\{schedule_label}"
        return scheduled_data, iters_count
                        
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
            'bert_input': scheduled_data['bert_input'].clone(),
            'bert_label': torch.zeros(scheduled_data['bert_label'].size(), dtype=scheduled_data['bert_label'].dtype),
            # 'segment_label': scheduled_data['segment_label'].clone().to(scheduled_data['segment_label'].dtype),
            'is_positive': scheduled_data['is_positive']
        }
        for b in range(scheduled_data['schedule_label'].size(0)):
            history_idx = ((scheduled_data['schedule_label'][b] > 0) & \
                (scheduled_data['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
            current_idx = ((scheduled_data['schedule_label'][b] > 0) & \
                (scheduled_data['schedule_label'][b] == order_label)).nonzero(as_tuple=True)[0]
            partial_data['bert_input'][b][history_idx] = scheduled_data['bert_label'][b][history_idx]
            partial_data['bert_label'][b][current_idx] = scheduled_data['bert_label'][b][current_idx]
            
        # print(f'\nk={order_label}, xo: \n{partial_data["bert_input"][0]}, xu: \n{partial_data["bert_label"][0]}')
        return partial_data        
        
        
    
    def iteration(self, epoch, data_loader, stage, train=True, schedule=None, \
                  is_ebm=True, visual_ebms=None):
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
        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP_%s_%s:%d" % (mode, stage, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )
        
        if self.sebm.task_name == 'countdown':
            special_tokens = {0,1,2,3,4}
            self.sebm.special_tok_size = len(special_tokens)
        else:
            raise NotImplementedError(f'special tokens for task {self.sebm.task_name} not specified!!')

        # for i, (pos_data, neg_data) in data_iter: #batch
        for i, data in data_iter: #batch
            if train:
                if stage == 'inference': #########for tesing
                    K = 10
                    t_list = unmasking_schedule(K+2, 'cosine')[1:-1]
                    scheduled_data, early_stop = self.add_schedule(data, t_list)
                else:
                    early_stop = 1
                for k in range(early_stop):
                    if stage == 'inference': #########for tesing
                        data = self.prepare_partial_data(scheduled_data, k+1) #AR-like
                    '''MLM training pradigm with varying t'''
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}
                    # neg_data = {key: value.to(self.device) for key, value in neg_data.items()}
                    if self.sebm.param_type == 'bert':
                        xo, xu = data['bert_input'], data['bert_label'] #already masked with rate t
                    elif self.sebm.param_type == 'gpt':
                        self.sebm.model.train()
                        # for param in self.sebm.model.parameters():
                        #     print(f'requires_grad: {param.requires_grad}')
                        xo, xu = data, data['labels'] #forward argument is a dict
                    # neg_xo, neg_xu = neg_data['bert_input'], neg_data['bert_label']
                    # print(f'\nxo({xo.shape}): \n{xo}\nxu({xu.shape}): \n{xu}\n') # both Size([4, 45])
                    # 1. Forward MLM to generate the gamma for calculating the energy landscapes
                    # gamma = self.sebm.forward(xo, data['segment_label'], is_ebm=True) 
                    gamma = self.sebm.forward(xo, None, is_ebm=True) 
                    # neg_gamma = self.sebm.model.forward(xo, neg_data['segment_label'], is_ebm=True) 
                    
                    #_________________test: BERT mlm_loss___________________
                    # print(f'\nxo.shape: {xo.shape}') #4,50
                    mlm_output, org_loss = self.sebm.forward(xo, None, is_ebm=False) #xo is partially unmasked 'bert_input'
                    # print(f'mlm_output.shape: {mlm_output.shape}') #Size(batch_size, |u'|, num_classes)
                    mlm_criterion = nn.CrossEntropyLoss()
                    mlm_loss = mlm_criterion(mlm_output.view(-1, mlm_output.size(-1)), xu.view(-1))
                    # print(f'\nmlm_loss: {mlm_loss}, loss: {org_loss}')
                    #———————————————————————————————————————————————————————
                    
                    # print(f'\ngamma shape: {gamma.shape}') #Size(batch_size, seq_len, num_classes)
                    
                    # _____________2-1 tentatively annotated for testing ___________
                    # 2-1. Estimate the 1D conditional logp(xu | xo) via pseudolikelihood
                    contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)
                    # recover the ignored token to padding
                    xu_ = xu.clone()
                    xu_[xu_ == -100] = 0
                    logp_xu_list, xu_list = [], [] #to store a batch of logp_xu's, of Size(batch_size, |u'|, num_classes)
                    for r in range(xu.size(0)): #iter within batch
                        # print(f'\n_____\nr: {r}, k: {k}\n________\n')
                        # assert torch.nonzero(xu[r]).squeeze().dim(), f"xu[r] is all zero: \n{xu[r]}"
                        if self.sebm.param_type == 'bert':
                            logp_xu = self.sebm.pseudolikelihood(gamma[r], xu_[r], xo[r]) #Size(|u'|, num_classes)
                        elif self.sebm.param_type == 'gpt':
                            logp_xu = self.sebm.pseudolikelihood(gamma[r], xu[r], xo['input_ids'][r]) #gamma[r]: Size(50, 31), xu[r]: Size(50), xo[r]: Size(50)
                        if (logp_xu == None):# or (logp_xu == -1): #not ebm #############################TODO: gpt需要加上后边的condition; bert要去掉后边的
                            continue #batch内该row fully unmasked, 不计算loss
                        xu_token_ids = torch.nonzero(xu_[r]).squeeze() #Size(|u'|)
                        if xu_token_ids.dim()==0 and xu_token_ids:
                            xu_token_ids = torch.tensor([xu_token_ids])
                        xu_label = xu[r][xu_token_ids]
                        assert logp_xu.size(0) == xu_label.size(0), \
                            f'\nlogp_xu.shape: {logp_xu.shape}, xu_label.shape: {xu_label.shape}'
                        logp_xu_list.append(logp_xu.unsqueeze(0))
                        xu_list.append(xu_label.unsqueeze(0))
                        # print(f'logp_xu.shape: {logp_xu.shape}, xu_label.shape({xu_label.shape}): {xu_label}')
                        # if r == 0:
                        #     print(f'xu_token_ids: {xu_token_ids}, \nxu_label: {xu_label}')
                        
                        # #___________sum over batch: contrast loss_______________
                        # contrast_loss = contrast_loss + self.sebm.calculate_contrast_loss(
                        #     gamma[r], gamma[r], #use the same latent
                        #     xu[r], neg_xu[r],
                        #     xo[r], neg_xo[r]
                        # )
                        # #_____________________________
                    #end of row iter
                    batch_logp_xu = torch.cat(logp_xu_list, dim=1).squeeze(0) #flattened: Size(sum of |u'| within the batch, num_classes)
                    batch_xu = torch.cat(xu_list, dim=1).view(-1) #flattened: Size(sum of |u'| within the batch) 
                    assert batch_logp_xu.dim(), f'k = {k}, early_stop = {early_stop}, logp_xu_list is empty!: {logp_xu_list}, '
                    # print(f'\nbatch_logp_xu.shape:{batch_logp_xu.shape}, batch_xu.shape: {batch_xu.shape}')
                    # 2-2. Contrast-loss
                    ce_loss = self.criterion(batch_logp_xu, batch_xu)
                    loss_is_nan = torch.isnan(ce_loss.clone().detach()).any()
                    assert loss_is_nan == False, f'\nce_loss becomes NaN! '\
                        f'\nce_loss: {ce_loss}, is_nan: {loss_is_nan}\n' \
                        f'batch_logp_xu: \n{batch_logp_xu}, \nbatch_xu: \n{batch_xu}, \ngamma[r]: \n{gamma[r]}'
                            
                    loss = ce_loss + contrast_loss
                    # loss = mlm_loss
                    if self.train_wandb:
                        wandb.log({"l_ce": ce_loss, "l_contrast": contrast_loss, "loss": org_loss, "mlm_loss": mlm_loss}) #loss: loss
                    # #_________mlm__________
                    if i % 5 == 0:
                        print(f'i={i}, k={k}, ce_loss: {ce_loss}, mlm_loss: {mlm_loss}')
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
                    self.optim_schedule.zero_grad()
                    loss.backward()
                    # Clip gradients
                    max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(self.sebm.model.parameters(), max_norm)
                    self.optim_schedule.step_and_update_lr()

                    # next sentence prediction accuracy
                    # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
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
                #end of k (schedule) iter (if any)
                
            elif (not train) and stage == 'inference':
                '''inference with scheduled t and sequential EBMs sampling'''
                # 1. break down the sequence tokens according to the schedule
                scheduled_data, early_stop = self.add_schedule(data, schedule)
                # print(f'scheduled_data: \n{scheduled_data}')
                # print(f'\nspecial_token_size: {self.sebm.special_tok_size}, vocab_size: {self.sebm.vocab_size}')
                # 2. iterate through k EBMs, send input to device, and each EBM performs gibbs sampling
                partial_pred = data['bert_input'].clone()
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
                    partial_pred, sth = self.sebm.sampling( #sth: loss_list
                        k+1, #1-indexed unmasking order label
                        partial_pred,
                        scheduled_data, 
                        self.sampler,
                        self.sampling_times,
                        visual_ebms=visual_ebms,
                    )
                    # print(f'{k}-th partial_pred: \n{partial_pred},\nloss_list: {sth}')
                    if visual_ebms == None:
                        k_losses[str(k)], k_energies[str(k)] = sth['losses'], sth['energies']
                    if k+1 == early_stop: #fully unmasked before reaching k
                        break
                pred = partial_pred
                invalid_pos = scheduled_data['bert_label'] < len(special_tokens)
                pred[invalid_pos] = scheduled_data['bert_label'][invalid_pos]
                correct = self.eval_metric(pred, scheduled_data['bert_label'])
                total_correct += correct
                total_samples = (i+1)*scheduled_data['bert_input'].size(0)
                # print(f"pred: \n{pred}, \nlabel: \n{scheduled_data['bert_label']},\ncorrrect: {correct}")
                # ______decode______
                label = scheduled_data['bert_label'][0].clone()
                label[label == -100] = 0
                pred[0][pred[0] == -100] = 0
                decoded_label = self.sebm.tokenizer.decode(label.tolist(), skip_special_tokens=True)
                decoded_pred = self.sebm.tokenizer.decode(pred[0].tolist(), skip_special_tokens=True)
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
                # print(f'data: \n{data}')
                if self.sebm.param_type == 'bert':
                    logits, _ = self.sebm.forward(data['bert_input'], None, is_ebm=False) # data['segment_label']
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data['bert_label'].view(-1)) #flattened (batch_size * seq_len)
                    # print(f'logits({logits.shape}): \n{logits}')
                    pred = logits.argmax(dim=-1) #argmin for "_mlm_test.pth"(训反了, energy漏加-); argmax for "_w_mask.pth"?
                    invalid_pos = data['bert_label'] < len(special_tokens)
                    pred[invalid_pos] = data['bert_label'][invalid_pos]
                    # print(f"logits.shape: {logits.shape}, label.shape: {data['bert_label'].shape}")
                    correct = self.eval_metric(pred, data['bert_label'])
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
                    total_samples += data['bert_input'].size(0)
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
                    stats['label'] = data['bert_label'].cpu().tolist()[0]
                elif self.sebm.param_type == 'gpt':
                    stats['label'] = data['labels'].cpu().tolist()[0]
                with open(eval_path, 'a') as statsfile:
                    statsfile.write(json.dumps(stats)+'\n')
                
            
            else:
                '''test the MLM training effectiveness and sampling'''
                # TODO 或许没用？
                raise NotImplementedError

            # if i % self.log_freq == 0: ###############
            #     data_iter.write(str(post_fix)) ##################
                
            break ###############test
            # if i == 10:
            #     break ###############test
        #end of batch iter
        
        # print(
        #     f"EP{epoch}, {mode}: \
        #     avg_loss={avg_loss / len(data_iter)}, \
        #     total_acc={total_correct * 100.0 / total_samples}"
        # ) 
        if train:
            print(f'\n\nFinished training.')
            ckpts_path = f'./ire_reasoning/ebm_ckpts/{self.sebm.task_name}_{self.sebm.param_type}{self.sebm.d_model}_{stage}_ebm.pth' #w_mask_inverse_test
            torch.save(self.sebm.model.state_dict(), ckpts_path)
            print(f'\nmodel saved to {ckpts_path}')
        elif (not train) and (schedule is not None):
            final_acc = round(total_correct*100/total_samples, 2)
            print(f'\nFinished sampling (inference), '\
                f'final accuracy: {final_acc}')
            with open(eval_path, 'a') as statsfile:
                statsfile.write(f'\nFinal Accuracy: {final_acc}\n')
            print(f'\nStats written to path: {eval_path}')
                
                
                
                
                
                
                

@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config):
    print(f'\n___\nStage: {config.train.stage}\n___\n')
    '''1. Load task datasets'''
    if config.param_type == 'bert':
        if config.task_name.startswith('binary'):
            max_len = config.tasks[config.task_name].inp_len + config.tasks[config.task_name].out_len #+ 3
        elif config.task_name == 'countdown':
            max_len = config.tasks[config.task_name].max_len
        train_loader, test_loader, train_size, test_size, tokenizer = load_bert_data(
            config.task_name, 
            'inference',#config.train.stage, #这里声明了stage: pretrain / sft
            max_len,
            config.train.batch_size, 
            config.sampling.batch_size
        )
    elif config.param_type == 'mlp':
        train_loader, test_loader, train_size, test_size = load_data(
            config.task_name, config.train.batch_size, config.sampling.batch_size) 

    elif config.param_type == 'gpt':
        max_len = config.models['gpt'].max_len
        train_loader, test_loader, train_size, test_size = load_gpt_data(
            config.task_name,
            max_len,
            config.train.batch_size,
            config.sampling.batch_size,
        )
    print(f'param type: {config.param_type}')
    print(f'\nLoaded datasets for task: {config.task_name}, max_len: {max_len}, ' \
        f'train:test={train_size}:{test_size},'\
        f'batch_size={config.train.batch_size}:{config.sampling.batch_size}...')
    # return ##############

    '''2. Initialize and train EBMs'''
    print(f'\nInitializing EBMs...')
    task_config = config.tasks[config.task_name]
    model_config = config.models[config.param_type]
    if task_config.name.startswith('binary'):
        print(f'inp_len: {task_config.inp_len}, out_len: {task_config.out_len}, num_classes: {task_config.num_classes}')
    if config.param_type == 'bert':
        d_model = config.models[config.param_type].d_model
        n_layers = config.models[config.param_type].n_layers
        heads = config.models[config.param_type].heads
        sebm = BERTSequentialEBMs(
            tokenizer=tokenizer,
            task_config=task_config,
            d_model=d_model, #######32 hidden_size
            n_layers=n_layers,
            heads=heads,
            device=config.device
            )
    elif config.param_type == 'gpt': #gpt2-6m-scratch from diffu-vs-ar paper
        gpt_config = AutoConfig.from_pretrained('./ire_reasoning/models/model_config_tiny') #pwd: EBM
        gpt2_scratch = AutoModelForCausalLM.from_config(gpt_config) #not ebm!
        sebm = GPTSequentialEBMs(
            gpt2_scratch,
            task_config,
            model_config,
            device=config.device, #TODO
        )
    else:
        raise NotImplementedError
    sebm_trainer = SequentialEBMsTrainer(
        sebm,
        train_loader,
        test_loader,
        config.train.lr,
        train_wandb=config.train.wandb,
        sampler=config.sampling.sampler,
        test_wandb=config.sampling.wandb,
        is_ebm=config.is_ebm,
        sampling_times=config.sampling.times,
        device=config.device
    )

    # return ##############

    if not config.load_ebm_ckpts:
        print(f'No checkpoints found.')
        # print(f'\nBefore training...')
        # # test(sebm, val_data[0]) 
        # k=10
        # sebm_trainer.evaluate(k, stage=config.sampling.stage, visualize=False)
        
        # return##############
        
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
        for epoch in range(config.train.epochs):
            sebm_trainer.train(epoch, config.train.stage)
        
        # return##############
    else:
        # ckpts_path = f'./ebm_ckpts/{task_config.name}_{config.param_type}' \
        #     f'{config.models[config.param_type].d_model}_diffusion.pth' # _w_mask_inverse.pth # _mlm_test # _w_mask #_diffusion
        # ckpts_path = './ire_reasoning/ebm_ckpts/countdown_bert384_sft.pth'
        ckpts_path = f'./ire_reasoning/ebm_ckpts/{task_config.name}_{config.param_type}{config.models[config.param_type].d_model}_{config.train.stage}.pth'
        print(f'\n3. Loading checkpoints from {ckpts_path}...')
        sebm_trainer.load_model(ckpts_path)##########################
        
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
    sebm_trainer.evaluate(k, stage=config.sampling.stage, visualize=False)


# Example usage
if __name__ == "__main__":
    main()