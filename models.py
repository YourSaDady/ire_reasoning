import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import os
import time
sys.path.append('/home/user/shiqi/yichuan/EBM/ire_reasoning')
os.chdir('/home/user/shiqi/yichuan/EBM/ire_reasoning')
# print(f'The current working directory: {os.getcwd()}')
from utils import convert_time

class ConditionalEBM(nn.Module):
    """MLP to model conditional distributions"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class SequentialEBM_old:
    def __init__(self, num_vars=5, num_classes=10, parameterization='mlp'):
        self.num_vars = num_vars
        self.num_classes = num_classes
        self.parameterization=parameterization
        self.models = self._build_model_sequence(parameterization)
    
    def _build_model_sequence(self, parameterization):
        models = {}
        # Build models for each conditional in the sequence
        for k in range(1, self.num_vars+1):
            if parameterization == 'mlp':
                models[k] = ConditionalEBM((k-1)*self.num_classes, self.num_classes)
            else:
                raise NotImplementedError(f'The EBM parameterization type: {parameterization} does not exist!!')
        return models
    
    '''
    Training with Cross Entropy and pseudolikelihood for each EBM
    '''
    def train_pseudolikelihood(self, data_batches, train_config, task_config):
        train_start = time.time()
        epochs = train_config.epochs
        lr = train_config.lr
        # Initialize model and optimizer
        model = self.models
        optimizers = {}
        for k in range(1, self.num_vars+1):
            optimizers[k] = optim.Adam(model[k].parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()
        #TODO: Data shape: (batch_size, num_vars)
        batch_size = train_config.batch_size
        sample_num = task_config.train_size
        print(f'\nbatch_size: {batch_size}, sample_num: {sample_num}')
        print(f'batch.shape: {data_batches[0].shape}')
        #TODO: iterate through all batches
        pbar = tqdm(total=sample_num)
        for bi, data in enumerate(data_batches):
            for k in range(1, self.num_vars+1):
                print(f"Training the EBM for first {k} variables...")
                k_vars_data = data[:, :k]

                for epoch in range(epochs):
                    optimizers[k].zero_grad()
                    # pseudo_logits = torch.ones(k_vars_data.shape[0], self.num_classes) #Size(batch_size, num_classes)
                    for var_idx in range(k):
                        # Get all other variable indices
                        other_vars = [i for i in range(k) if i != var_idx]
                        # Create conditional dataset
                        inputs = k_vars_data[:, other_vars]
                        targets = k_vars_data[:, var_idx] #batchalized one-hot vector
                        # Convert inputs to one-hot
                        try:
                            inputs_onehot = torch.zeros(inputs.shape[0], (k-1)*self.num_classes)
                            for b in range(inputs.shape[0]):
                                inputs_onehot[b] = torch.cat([
                                    nn.functional.one_hot(inputs[b,i].long(), self.num_classes).float()
                                    for i in range(k-1)
                                ])
                        except: #unconditional p(x1)
                            inputs_onehot = torch.zeros(batch_size, 0)
                        logits = model[k](inputs_onehot)
                        # assert pseudo_logits.shape == logits.shape
                        #calculate the pseudolikelihood by taking product of all conditional logits
                        # pseudo_logits = torch.mul(pseudo_logits, logits) 
                        loss = criterion(logits, targets.long())
                        loss.backward()
                        optimizers[k].step()
                    #end of var_idx iter
                    if epoch % 20 == 0:
                        print(f"Batch={bi}, K={k}, Var={var_idx+1}, Epoch {epoch}: Loss {loss.item():.4f}")
                #end of epoch iter
            #end of ebm iter
            pbar.update(batch_size)
            # break#########################
        pbar.close()
        #end of batch iter
        train_spent = convert_time(train_start)
        print(f'\nFinished training in {train_spent[0]}:{train_spent[1]}:{train_spent[2]}')

    '''
    Inference (Gibbs-sampling)
    '''
    def gibbs_sample(self, observed, steps=1000, temperature=1.0): #not batchlized?
        """Gibbs sampling for variables completion"""
        sample = torch.zeros(self.num_vars)
        sample[:len(observed)] = torch.tensor(observed)
        
        for _ in range(steps):
            for i in range(len(observed), self.num_vars): # iterate through unobserved positions
                # Get current context
                context = sample[[j for j in range(self.num_vars) if j != i]] #other vars (含observed) all time t
                # Convert context to one-hot
                context_onehot = torch.cat([
                    nn.functional.one_hot(context[j].long(), self.num_classes).float() # time t
                    for j in range(self.num_vars-1)
                ])
                # Get conditional distribution
                with torch.no_grad():
                    logits = self.models[i+1](context_onehot)
                # Sample new value
                probs = nn.functional.softmax(logits/temperature, dim=-1) #normalize?
                sample[i] = torch.multinomial(probs, 1).float()
                
        return sample.numpy().astype(int)[len(observed):]

    def save_ckpts(self, task_config):
        ebms_save_prefix = f'./ebm_ckpts/{task_config.name}'
        os.makedirs(ebms_save_prefix, exist_ok=True)
        for ebm_idx in self.models:
            ckpt_save_path = f'{ebms_save_prefix}/{self.parameterization}_{ebm_idx}.pth'
            torch.save(self.models[ebm_idx].state_dict(), ckpt_save_path)
        print(f"\nAll EBMs' state_dict are saved to {ebms_save_prefix}")
    
    def load_ckpts(self, ckpts_prefix):
        for ebm_idx in self.models:
            self.models[ebm_idx].load_state_dict(
                torch.load(f'{ckpts_prefix}/{self.parameterization}_{ebm_idx}.pth')
            )
