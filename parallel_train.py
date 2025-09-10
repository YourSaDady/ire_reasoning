'''
Multi-GPUs Parallel version of main.py
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
import transformers 
from transformers import AutoConfig
# print(f'The current working directory: {os.getcwd()}')
import hydra
from sequential_ebms import BERTSequentialEBMs, GPTSequentialEBMs, FastSequentialEBMs
from main import unmasking_schedule, ScheduledOptim, SequentialEBMsTrainer, EarlyStopper
from dataset import load_data
from utils import convert_time, VisualizeEBMs
import random as rand
from time import time
import json
import math

IGNORE_INDEX = -100

# parallel-related
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12374'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
    
class ParallelSequentialEBMsTrainer(SequentialEBMsTrainer):
    def __init__(
        self,
        sebm,
        train_dataloader,
        test_dataloader,
        lr=1e-4,
        # weight_decay=0.01,
        # betas=(0.9, 0.999),
        # warmup_steps=10000,
        # log_freq=10,
        train_wandb=True,
        test_wandb=False,
        train_size=-1,
        test_size=-1,
        sampler='gibbs',
        sampling_times=10,
        is_ebm=True,
        contrast=False,
        device=None,
        epochs=1,
        continue_train=False,
        
    ):
        super().__init__(sebm, 
            train_dataloader, 
            test_dataloader, 
            lr,
            sampler=sampler,
            train_wandb=train_wandb,
            test_wandb=test_wandb,
            train_size=train_size,
            test_size=test_size,
            is_ebm=is_ebm,
            contrast=contrast,
            sampling_times=sampling_times,
            device=device,
            parallel=True,
            epochs=epochs,
            continue_train=continue_train,
        )
        
        # Wrap the model with DistributedDataParallel
        self.sebm.model = DDP(self.sebm.model, \
            device_ids=[self.device])


'''
Parallel version of main()
'''
def train(rank, world_size, config):
    os.chdir(config.project_root)
    '''0. Set up parallel environment'''
    setup(rank, world_size)
    print(f'rank: {rank}, pid: {os.getpid()}')
    '''1. Load task datasets'''
    if config.task_name.startswith('cd') or config.task_name == 'sudoku':
        max_len = config.tasks[config.task_name].max_len
    else:
        raise NotImplementedError(f'{config.task_name} is not specified!')
    train_loader, test_loader, train_size, test_size, tokenizer = load_data(
        config.task_name, 
        stage='inference',#config.train.stage, #这里声明了stage: pretrain / sft
        max_len=max_len,
        train_batch_size=config.train.batch_size, 
        val_batch_size=config.sampling.batch_size,
        contrast=config.train.contrast,
        parallel=config.parallel,
    )
    print(f'param type: {config.param_type}\ntrain_batch_size:test_batch_size={config.train.batch_size}:{config.sampling.batch_size}')
    print(f'\nLoaded datasets for task: {config.task_name}, max_len: {max_len}, ' \
        f'train:test={train_size}:{test_size} (before batching)...')
    print(rank, "local BS =", next(iter(train_loader))['input'].shape[0])
    # return ##############
    
    '''2. Initialize sequential EBMs'''
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
            device=rank
            )
        # assert all(param.device.type == rank for param in sebm.model.parameters()), \
        #     f'{[param.device.type for param in sebm.model.parameters()]}'
    elif config.param_type == 'gpt': #gpt2-6m-scratch from diffu-vs-ar paper
        sebm = GPTSequentialEBMs( #flexible input and output length
            tokenizer,
            task_config,
            model_config=None, #TODO: other configs, if necessary
            device=rank,
        )
    elif config.param_type == 'fast':
        model_config_path = f'./models/model_config_{config.model_scale}'
        model_config = AutoConfig.from_pretrained(model_config_path)
        sebm = FastSequentialEBMs(
            tokenizer=tokenizer,
            task_config=task_config,
            model_config=model_config,
            model_arc=config.model_arc,
            model_scale=config.model_scale,
            device=rank,
        )
    else:
        raise NotImplementedError
    
    '''3. Initialize Trainer'''
    sebm_trainer = ParallelSequentialEBMsTrainer(
        sebm, 
        train_loader, 
        test_loader, 
        lr=config.train.lr,
        train_size=train_size, #the total number of trai samples, not batch size
        test_size=test_size, #similar
        train_wandb=config.train.wandb,
        sampler=config.sampling.sampler,
        test_wandb=config.sampling.wandb,
        is_ebm=config.is_ebm,
        contrast=config.train.contrast,
        sampling_times=config.sampling.times,
        device=rank,
        epochs=config.train.epochs,
        continue_train=config.continue_train,
    )
    
    if config.train.wandb and sebm_trainer.device == 0:
        print(f'\n\n\nStart training and initialize wandb...')
        import wandb
        # wandb.login()
        run = wandb.init(
            project=f'EBM_train-{task_config.name}_{config.param_type}_rank0',
            config={ # Track hyperparameters and metadata
                "learning_rate": config.train.lr,
                "epochs": config.train.epochs,
            },
        )
        
    '''0. Set the early_stopper'''
    early_stopper = EarlyStopper(patience=25, min_delta=5e-4, ema_beta=0.9, mode='min')
        
    for epoch in range(config.train.epochs):
        if config.continue_train and epoch < config.cont_epoch:
            continue
        elif config.continue_train and epoch == config.cont_epoch:
            n_steps = config.cont_epoch * train_size
            sebm_trainer.optim_schedule.fast_forward(n_steps)
            print(f'device{sebm_trainer.device} have optimizer set to the latest step, start training from epoch{epoch}')
        # dist.barrier()
        # print(f'device{sebm_trainer.device} synchronized')
        '''1. train'''
        sebm_trainer.train_data.sampler.set_epoch(epoch) #make shuffling work properly across multiple epochs
        sebm_trainer.train(epoch, config.train.stage)
        # if sebm_trainer.train_is_converged: #TODO: remove
        #     print(f'\nend epochs loop')
        #     break
        
        '''2. validate converge or not'''
        print(f'device{sebm_trainer.device}: epoch{epoch} finished, waiting for sync...')
        # dist.barrier()
        # print(f'device{sebm_trainer.device} synchronized')
        converge, val_acc, val_ce = sebm_trainer.validate(epoch, early_stopper)
        
        # if val_acc >= 98.0:
        #     torch.save(sebm_trainer.sebm.model.state_dict(), sebm_trainer.ckpts_path)
        #     print(f'\n\nval_acc: {val_acc} is over 98%, saved to {sebm_trainer.ckpts_path}\n\n')
        #     break
        if config.train.wandb and sebm_trainer.device == 0:
            # print(f'epoch{epoch}: val_ce: {val_ce}, val_acc: {val_acc}')
            wandb.log({'val_ce': val_ce, 'val_acc': val_acc})
        if converge or (epoch == sebm_trainer.epochs-1):
            print(f'\n\n你converged!!\ndevice: {sebm_trainer.device}\nepoch: {epoch}\nval_acc: {val_acc}\nval_ce: {val_ce}\n\n')
            break
        else:
            print(f'\n\nNot convreged...\ndevice: {sebm_trainer.device}\nepoch: {epoch}\nval_acc: {val_acc}\nval_ce: {val_ce}\n\n')
    
    cleanup()


@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(config):
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    print(f'start spawning!')
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True) 


if __name__ == "__main__":
    main()
    

