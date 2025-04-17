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
from tqdm import tqdm
sys.path.append('/home/user/shiqi/yichuan/EBM/ire_reasoning')
os.chdir('/home/user/shiqi/yichuan/EBM/ire_reasoning')
print(f'The current working directory: {os.getcwd()}')
import hydra
from models import SequentialEBM
from datasets import load_bert_data

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

def SequentialEBMsTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader,
        stage,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warnup_steps=10000,
        log_freq=10, #log every 10 batches
        device='cpu' ########debugging
    ):
        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.stage = stage
        # Schedule the optimizer (as stated in paper)
        self.optim = optim.AdamW(self.model.bert.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
        )
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.bert.parameters()]))
        
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
        
        
    def iteration(self, epoch, data_loader, train=True): #algorithm core function
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            print(f'\ndata: {data}')
            
            
            xo, xu = data['bert_input'], data['bert_label'] #already masked with rate t
            print(f'\nxo({xo.shape}): \n{xo}\nxu({xu.shape}): \n{xu.shape}\n')
            # 1. Forward MLM to generate the gamma for calculating the energy landscapes
            gamma = self.model.forward(xo, data['bert_label'], is_ebm=True) 
            print(f'\ngamma shape: {gamma.shape}') #Size(batch_size, out_len, num_classes)
            
            # 2-1. Estimate the 1D conditional logp(xu | xo) via pseudolikelihood
            ce_loss = torch.tensor(0., requires_grad=True).to(self.device)
            for r in range(xu.size(0)): #iter through batch
                logp_xu = self.model.pseudolikelihood(gamma[r], xu[r]) #Size(|u'|, num_classes)
                xu_label = torch.nonzero(xu[r]).squeeze() #Size(|u'|)
                assert lop_xu.size(0) == xu_label.size(0), \
                    f'\nlop_xu.shape: {logp_xu.shape}, xu_label.shape: {xu_label.shape}'
                ce_loss += self.criterion(logp_xu, xu_label)
            
            # 2-2. Contrast-loss
            # TODO
            contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)#########暫设0
            loss = ce_loss + contrast_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                # "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}, \
            total_acc={total_correct * 100.0 / total_element}"
        ) 
    

@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config):
    '''1. Load task datasets'''
    print(f'\nLoading datasets for task: {config.task_name}, max_len: {config.max_len}...')
    max_len = config.tasks[config.task_name].inp_len + config.tasks[config.task_name].out_len
    train_loader, test_loader, train_size, test_size = load_bert_data(config.task_name, config.train.stage, max_len, config.train.batch_size, config.sampling.batch_size) 

    # return ##############

    '''2. Initialize and train EBMs'''
    print(f'\nInitializing EBMs...')
    task_config = config.tasks[config.task_name]
    print(f'inp_len: {task_config.inp_len}, out_len: {task_config.out_len}, num_classes: {task_config.num_classes}')
    sebm = SequentialEBM(
        parameterization=config.param_type,
        task_config=task_config,
        special_tokens=config.special_tokens,
        )
    sebm_trainer = SequentialEBMsTrainer(
        sebm,
        train_loader,
        test_loader,
        stage=config.train.stage,
        device='cpu'
    )

    # return ##############

    if not config.load_ebm_ckpts:
        print(f'No checkpoints found.')
        print(f'\nBefore training...')
        # test(sebm, val_data[0]) 
        sebm_trainer.evaluate(val_data, store_stat=False, sampling_config=config.sampling, visual_config=config.visualize) 
        
        # return##############
        
        print(f'\n\n\n3. Start training...')
        sebm_trainer.train(train_data, config.train, config.tasks[config.task_name], config.visualize)
        
        # return##############
    else:
        print(f'\n3. Loading checkpoints...')
        sebm.load_ckpts()

    '''3. Evaluate'''
    print(f'\n\n\n4. Start evaluation...')
    sebm_trainer.evaluate(val_data, store_stat=True, sampling_config=config.sampling, visual_config=config.visualize)


# Example usage
if __name__ == "__main__":
    main()