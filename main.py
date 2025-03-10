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

@hydra.main(version_base=None, config_path='./configs',
            config_name='config')
def main(config):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='binary-addition')
    # parser.add_argument('--param_type', type=str, default='mlp')
    # args = parser.parse_args()

    # Generate synthetic training data (replace with real Sudoku data)
    # num_samples = 1000
    # data = torch.randint(0, 10, (num_samples, 5)).float()
    '''1. Load task datasets'''
    print(f'Loading datasets for task: {config.task}...')
    train_data, val_data, train_size, val_size = load_data(config.task, config.train.batch_size) #TODO: batchalized
    
    '''2. Initialize and train EBMs'''
    print(f'Initializing EBMs...\nnum_vars={config.tasks[config.task].seq_len}\nnum_classes={config.tasks[config.task].vol}')
    sebm = SequentialEBM(num_vars=config.tasks[config.task].seq_len, \
        num_classes=config.tasks[config.task].vol, \
        parameterization=config.param_type)
    if not config.load_ebm_ckpts:
        print(f'\nTraining...')
        sebm.train_pseudolikelihood(train_data, config.train, config.tasks[config.task])
        sebm.save_ckpts(config.tasks[config.task])
    else:
        ckpts_prefix = f'./ebm_ckpts/{config.task}'
        print(f'\n Loading existing EBMs ckpts at: {ckpts_prefix}...')
        sebm.load_ckpts(ckpts_prefix)
    '''3. Evaluate'''
    print(f'\nStart evaluation...')
    stat_prefix = f'./stats/{config.task}/{config.param_type}'
    os.makedirs(stat_prefix, exist_ok=True)
    stat_path = f'{stat_prefix}/evaluate_{val_size}.jsonl'
    with open(stat_path, 'w') as stat_file:
        acc_count = 0
        for id, (observed, answer) in enumerate(val_data):
            completed = sebm.gibbs_sample(observed, \
                steps=config.sample.steps, \
                temperature=config.sample.temperature)
            correct = np.array_equal(completed, answer.numpy().astype(int))
            if correct:
                acc_count += 1
            stat = {
                'id': id,
                'correct': correct,
                'observed': observed.numpy().tolist(),
                'completed': completed.tolist(),
            }
            print(stat)
            stat_file.write(json.dumps(stat)+'\n')
        #end of sample iter
    final = {'acc': acc_count/val_size}
    stat_file.write('\n'+json.dumps(final))
    print(f'\nEvaluation Result: {final}')
    #end of write stat

# Example usage
if __name__ == "__main__":
    main()