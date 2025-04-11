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
from datasets import load_data

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


@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config):
    '''1. Load task datasets'''
    print(f'\nLoading datasets for task: {config.task_name}...')
    train_data, val_data, train_size, val_size = load_data(config.task_name, config.train.batch_size, config.sampling.batch_size) 

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

    # return ##############

    if not config.load_ebm_ckpts:
        print(f'No checkpoints found.')
        print(f'\nBefore training...')
        # test(sebm, val_data[0]) 
        sebm.evaluate(val_data, store_stat=False, sampling_config=config.sampling, visual_config=config.visualize) 
        
        # return##############
        
        print(f'\n\n\n3. Start training...')
        sebm.train(train_data, config.train, config.tasks[config.task_name], config.visualize)
        
        # return##############
    else:
        print(f'\n3. Loading checkpoints...')
        sebm.load_ckpts()

    '''3. Evaluate'''
    print(f'\n\n\n4. Start evaluation...')
    sebm.evaluate(val_data, store_stat=True, sampling_config=config.sampling, visual_config=config.visualize)


# Example usage
if __name__ == "__main__":
    main()