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
print(f'The current working directory: {os.getcwd()}')
import hydra
from sequential_ebms import BERTSequentialEBMs
from dataset import load_bert_data, load_data
from utils import convert_time, VisualizeEBMs
from transformers import BertTokenizer
import random as rand
import wandb

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
        wandb=True,
        device='cpu' ########debugging
    ):
        self.device = device
        self.sebm = sebm
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        # Schedule the optimizer (as stated in paper)
        self.optim = optim.AdamW(self.sebm.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.sebm.d_model, n_warmup_steps=warmup_steps
        )
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.log_freq = log_freq
        # sampling configs:
        self.sampler = sampler
        self.sampling_times = sampling_times
        self.wandb = wandb
        print("Total Parameters:", sum([p.nelement() for p in self.sebm.model.parameters()]))
    
    def load_model(self, ckpts_path):
        state_dict = torch.load(ckpts_path)
        self.sebm.model.load_state_dict(state_dict)
        
    def train(self, epoch, stage):
        self.iteration(epoch, self.train_data, stage)

    def test(self, epoch, stage):
        self.iteration(epoch, self.test_data, stage, train=False)
        
    def evaluate(self, k, scheduler='cosine', visualize=False): #Inference
        '''
        Recover a fully masked sequence using a specified scheduler, with decreasing t
        '''
        t_list = unmasking_schedule(k+2, scheduler)[1:-1]
        # print(f't_list: {t_list}')
        if visualize:
            raise NotImplementedError
        else:
            self.iteration(1, self.test_data, stage='inference', train=False, \
                schedule=t_list)
        
    def add_schedule(self, data, schedule):
        '''
        simply add a schedule label key-value pair to the data dict
        
        params:
            data: batchalized data dict with bert_inputs, bert_labels, segment_labels and is_positive
            schedule: a list of decreasing unmasking rate t
        return:
            scheduled_data: batchalized data dict extended with schedule_label k-v pair
        '''
        scheduled_data = data
        batch_size, io_len = data['bert_label'].size()
        print(f"bert_label.shape: {data['bert_label'].shape}")
        schedule_label = [[0]*io_len]*batch_size #initialize a 2D batchalized schedule label
        special_ids = {0,1,2,3}
        full_label = data['bert_label'].tolist()
        assert len(full_label)==batch_size and len(full_label[0])==io_len, \
            f'bert_label: Size({len(full_label)}, {len(full_label[0])})'
        for k, t in enumerate(schedule):
            unmask_num = int(t * torch.count_nonzero(data['bert_label'][0]))
            assert unmask_num > 0, f'unmasking number should be positive! Got {unmask_num}'
            print(f'k: {k}, t: {t}, unmask_num: {unmask_num}')
            for bid in range(batch_size):
                for pos, label in enumerate(full_label[bid]):
                    #pick the remaining t2 tokens as unmasking candidate
                    if label not in special_ids and schedule_label[bid][pos]==0: 
                        prob = rand.random()
                        if prob < t:
                            schedule_label[bid][pos] = k+1 #1-indexed
                    if torch.count_nonzero(schedule_label[bid]) == unmask_num:
                        break
        schedule_label = torch.tensor(schedule_label)
        print(f'\nschedule_label: \n{schedule_label}')
        scheduled_data['schedule_label'] = schedule_label
        assert torch.count_nonzero(schedule_label) == torch.count_nonzero(data['bert_label']), \
            f'nonzero labels count: schedule={torch.count_nonzero(schedule_label)}, ' \
                f"bert={torch.count_nonzero(data['bert_label'])}"
        return scheduled_data    
                        
    def eval_metric(self, pred, label, metric="acc"):
        '''Calcualte the evaluation metrics for a batch pair of predictions and labels'''
        if metric == 'acc':
            matching_rows = (pred == label).all(dim=1)  # Check for equality along the rows
            return matching_rows.sum().item()  # Sum up the True values
    
    def iteration(self, epoch, data_loader, stage, train=True, schedule=None, \
                  visual_ebms=None):
        '''
        Algorithm core function (exclude sampling details)
        Performs train / test / evaluation iterations during different stages
        '''
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP_%s_%s:%d" % (mode, stage, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )
        if (not train) and (schedule is not None):
            total_correct = 0

        # for i, (pos_data, neg_data) in data_iter: #batch
        for i, data in data_iter: #batch
            if train:
                '''MLM training pradigm with varying t'''
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}
                # neg_data = {key: value.to(self.device) for key, value in neg_data.items()}
                
                xo, xu = data['bert_input'], data['bert_label'] #already masked with rate t
                # neg_xo, neg_xu = neg_data['bert_input'], neg_data['bert_label']
                # print(f'\nxo({xo.shape}): \n{xo}\nxu({xu.shape}): \n{xu}\n') # both Size([4, 45])
                # 1. Forward MLM to generate the gamma for calculating the energy landscapes
                gamma = self.sebm.model.forward(xo, data['segment_label'], is_ebm=True) 
                # neg_gamma = self.sebm.model.forward(xo, neg_data['segment_label'], is_ebm=True) 
                
                # #_________________test: BERT mlm_loss___________________
                # mlm_output = self.sebm.model.forward(xo, data['segment_label'], is_ebm=False) #test, softmaxed
                # mlm_criterion = nn.NLLLoss(ignore_index=0)
                # mlm_loss = mlm_criterion(mlm_output.transpose(1, 2), xu)
                # #———————————————————————————————————————————————————————
                
                # print(f'\ngamma shape: {gamma.shape}') #Size(batch_size, seq_len, num_classes)
                
                # 2-1. Estimate the 1D conditional logp(xu | xo) via pseudolikelihood
                ce_loss = torch.tensor(0., requires_grad=True).to(self.device)
                contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)
                for r in range(xu.size(0)): #iter within batch
                    # assert torch.nonzero(xu[r]).squeeze().dim(), f"xu[r] is all zero: \n{xu[r]}"
                    logp_xu = self.sebm.pseudolikelihood(gamma[r], xu[r], xo[r]) #Size(|u'|, num_classes)
                    xu_token_ids = torch.nonzero(xu[r]).squeeze() #Size(|u'|)
                    if xu_token_ids.dim()==0 and xu_token_ids:
                        xu_token_ids = torch.tensor([xu_token_ids])
                    xu_label = xu[r][xu_token_ids]
                    assert logp_xu.size(0) == xu_label.size(0), \
                        f'\nlogp_xu.shape: {logp_xu.shape}, xu_label.shape: {xu_label.shape}'
                    # print(f'logp_xu.shape: {logp_xu.shape}, xu_label.shape({xu_label.shape}): {xu_label}')
                    
                    
                    # #___________sum over batch: contrast loss_______________
                    # contrast_loss = contrast_loss + self.sebm.calculate_contrast_loss(
                    #     gamma[r], gamma[r], #use the same latent
                    #     xu[r], neg_xu[r],
                    #     xo[r], neg_xo[r]
                    # )
                    # #_____________________________
                    
                    ce_loss = ce_loss + self.criterion(logp_xu, xu_label)
                    loss_is_nan = torch.isnan(torch.tensor(ce_loss)).any()
                    assert loss_is_nan == False, f'\nce_loss becomes NaN! '\
                        f'\nce_loss: {ce_loss}, is_nan: {loss_is_nan}\n' \
                        f'logp_xu: \n{logp_xu}, \nxu_label: \n{xu_label}, \ngamma[r]: \n{gamma[r]}'
                
                # 2-2. Contrast-loss
                # TODO
                loss = ce_loss + contrast_loss
                # loss = mlm_loss
                if self.wandb:
                    wandb.log({"l_ce": ce_loss, "l_contrast": contrast_loss, "loss": loss})

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
                # total_element += data["is_next"].nelement()
                post_fix = { # TODO: 不同情况k v 不同
                    "epoch": epoch,
                    "sample": i,
                    "avg_loss": avg_loss / (i + 1),
                    # "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }
                
            elif (not train) and (schedule is not None):
                '''inference with scheduled t and sequential EBMs sampling'''
                # 1. break down the sequence tokens according to the schedule
                scheduled_data = self.add_schedule(data, schedule)
                print(f'scheduled_data: \n{scheduled_data}')
                # 2. iterate through k EBMs, send input to device, and each EBM performs gibbs sampling
                partial_pred = torch.zeros_like(scheduled_data['bert_input']) #init
                k_losses = {}
                for k, t in enumerate(schedule):
                    partial_pred, sth = self.sebm.sampling(
                        k+1, #1-indexed unmasking order label
                        partial_pred,
                        scheduled_data, 
                        self.sampler,
                        self.sampling_times,
                        visual_ebms=visual_ebms
                    )
                    print(f'{k}-th partial_pred: \n{partial_pred}')
                    if visual_ebms == None:
                        k_losses[k] = sth
                pred = partial_pred
                correct_count = self.eval_metric(pred, scheduled_data['bert_label'])
                total_correct += correct_count
                total_samples = (i+1)*scheduled_data['bert_input'].size(0)
                print(f'\nfinal pred: \n{pred}, \nk_losses: \n{k_losses}')
                print(f"\nlabels: {scheduled_data['bert_label']}\ncorrect_count: {correct_count}")
                # 3. check correctness (tk-avg and final) and record the energy landscape TODO
                # TODO 单独记录energy变化
                post_fix = {
                    "sample": i,
                    "acc": round(total_correct*100/total_samples, 2)
                }
                if self.wandb:
                    wandb.log({'acc': "acc"})
            
            else:
                '''test the MLM training effectiveness and sampling'''
                # TODO 或许没用？
                raise NotImplementedError

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                
            # break ###############test
        #end of batch iter
        
        # print(
        #     f"EP{epoch}, {mode}: \
        #     avg_loss={avg_loss / len(data_iter)}, \
        #     total_acc={total_correct * 100.0 / total_element}"
        # ) 
        if train:
            print(f'\n\nFinished training.')
            ckpts_path = f'./ebm_ckpts/{self.sebm.task_name}_{self.sebm.param_type}{self.sebm.d_model}_mlm.pth'
            torch.save(self.sebm.model.state_dict(), ckpts_path)
            print(f'\nmodel saved to {ckpts_path}')
    

@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config):
    print(f'\n___\nStage: {config.train.stage}\n___\n')
    '''1. Load task datasets'''
    if config.param_type == 'bert':
        max_len = config.tasks[config.task_name].inp_len + config.tasks[config.task_name].out_len + 3
        train_loader, test_loader, train_size, test_size = load_bert_data(
            config.task_name, 
            config.train.stage, #这里声明了stage: pretrain / sft
            max_len,
            config.train.batch_size, 
            config.sampling.batch_size
        )
        print(f'\nLoaded datasets for task: {config.task_name}, max_len: {max_len}, ' \
            f'train:test={train_size}:{test_size},'\
            f'batch_size={config.train.batch_size}:{config.sampling.batch_size}...')
    elif config.param_type == 'mlp':
        train_loader, test_loader, train_size, test_size = load_data(
            config.task_name, config.train.batch_size, config.sampling.batch_size) 

    # return ##############

    '''2. Initialize and train EBMs'''
    print(f'\nInitializing EBMs...')
    task_config = config.tasks[config.task_name]
    print(f'inp_len: {task_config.inp_len}, out_len: {task_config.out_len}, num_classes: {task_config.num_classes}')
    if config.param_type == 'bert':
        sebm = BERTSequentialEBMs(
            task_config=task_config,
            d_model=config.d_model, #######32 hidden_size
            device='cpu'
            )
    else:
        raise NotImplementedError
    sebm_trainer = SequentialEBMsTrainer(
        sebm,
        train_loader,
        test_loader,
        config.train.lr,
        wandb=config.train.wandb
    )

    # return ##############

    if not config.load_ebm_ckpts:
        # print(f'No checkpoints found.')
        # print(f'\nBefore training...')
        # # test(sebm, val_data[0]) 
        # sebm_trainer.evaluate(1, config.train.stage) 
        
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
        print(f'\n3. Loading checkpoints...')
        ckpts_path = f'./ebm_ckpts/{task_config.name}_{config.param_type}.pth'
        sebm_trainer.load_model(ckpts_path)
        

    return ###############

    '''3. Evaluate'''
    k = 10
    if config.train.wandb:
        wandb.login()
        run = wandb.init(
            project=f'EBM_eval-{task_config.name}_{config.param_type}',  # Specify your project
            config={                        # Track hyperparameters and metadata
                "k": k,
                "batch_size": config.sampling.batch_size,
            },
        )
    print(f'\n\n\n4. Start evaluation...')
    sebm_trainer.evaluate(k, visualize=False)


# Example usage
if __name__ == "__main__":
    main()