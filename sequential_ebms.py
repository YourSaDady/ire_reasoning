'''
Seqential EBMs wrappers for all base models
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import os
import os.path as osp
import time
import random
import json
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from dash import Dash, dcc, html
sys.path.append('/home/yichuan/HKU/EBM/ire_reasoning')
os.chdir('/home/yichuan/HKU/EBM/ire_reasoning')
os.environ['WANDB_API_KEY'] = '3c06642500f1527ecd0328870ff61d36b5c17193'
# os.environ['CUDA_LAUNCH_BLOCKING']=1
# os.environ['TORCH_USE_CUDA_DSA']=1
# print(f'The current working directory: {os.getcwd()}')
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
# from transformers.modeling_outputs import CausalLMOutput
from models.custom_tokenizer import CustomTokenizer
from models.bert import PositionalEmbedding

from utils import convert_time, VisualizeEBMs, check_grad
from typing import Optional, Union, Callable
import wandb

from models.mlp import MLP
from models.bert import DiscreteDiffusion, EnergyBERT

inf = 1000000
IGNORE_INDEX = -100

def swish(x): #当做一种mlp的activation就好：在linear与ReLU之间可调。beta=1的时候(i.e. this definition)叫SiLU (Sigmoid Linear Unit)
    return x * torch.sigmoid(x)

# def shuffle(index_list):
#     snh48 = inde_list.copy()
#     snh48 = random.shuffle(snh48)
#     return snh48

'''Randomly change p-rate of the elements in a tensor within the value range'''
def random_flip(samples: Union[torch.Tensor, list[torch.Tensor]], flip_range:int, rate=0.5) -> Union[torch.Tensor, list[torch.Tensor]]:
    if isinstance(samples, torch.Tensor):
        samples = [samples]
    flipped_samples = []
    for sample in samples:
        batch_size, seq_len = sample.size()
        flip_len = max(int(rate * seq_len), 1) #flip when seq_len=1

        # Generate a mask to randomly select columns to flip
        flip_mask = torch.zeros(seq_len)
        flip_mask[:flip_len] = 1
        flip_mask = flip_mask[torch.randperm(seq_len)]

        # Apply the mask to flip the selected columns for all rows in the batch
        flipped_tensor = sample.clone()
        for i in range(batch_size):
            flipped_tensor[i] = torch.where(flip_mask.bool(), 1 - flipped_tensor[i], flipped_tensor[i])
        flipped_samples.append(flipped_tensor)
    
    if len(samples)==1:
        return flipped_samples[0]
    return flipped_samples

'''Disabled'''
# class MLPSequentialEBMs():
#     '''
#     sequential EBMs wrapper for MLP model architectures
#     '''
#     def __init__(self, parameterization='mlp', task_config=None, special_tokens=['<mask>']):
#         self.task_config=task_config
#         self.inp_len = task_config.inp_len
#         self.out_len = task_config.out_len
#         self.ebm_num = self.out_len #assume the number of EBMs equals the total number of variables
#         self.num_classes = task_config.num_classes #exclude the special tokens, eg. <mask>
#         self.metrics = task_config.metrics #list of metric names
#         #build vocabulary dictionary (include special tokens)
#         self.vocab = {v_id:str(v_id) for v_id in range(self.num_classes)}
#         self.special_tokens = special_tokens
#         for s_id in range(len(special_tokens)):
#             self.vocab[self.num_classes+s_id] = special_tokens[s_id]

#         self.parameterization = parameterization
#         self.model = self._build_model(self.parameterization)
#         self.generate_call_count = 0
#         self.device= 'cpu' #'cuda'
        
#     def _build_model(self, parameterization):
#         if parameterization == 'mlp':
#             return MLP(self.inp_len, self.out_len, self.num_classes)
#         elif parameterization == 'bert':
#             return DiscreteDiffusion(vocab_size=self.num_classes) #x.shape = Size(batch_size, 1, seq_len, seq_len)
#         else:
#             raise NotImplementedError
        

#     def mask(self, x: torch.Tensor, y: torch.Tensor, mask_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
#         '''
#         randomly mask y, concatenate x with the masked y as observable x_o.
        
#         params: 
#             x: Size(batch_size, inp_len)
#             y: Size(batch_size, out_len)
#             mask_idx: the index representing the <MASK> state
            
#         returns:
#             xo: Size(batch_size, inp_len+out_len), obervables, input to the base model
#             xu: Size(batch_size, |u|), unobservables, output to be unmasked
#             u: the list of masked indices
#         '''
#         out_len = y.size(1)
#         num_masks = random.randint(1, out_len)  # Randomly choose the number of masks (normally dist)
#         u = random.sample(range(out_len), num_masks)  # Randomly sample index positions for masks (already shuffled!)
#         mask_list = [1 if i in u else 0 for i in range(out_len)] #all samples in a batch share the same masking indices
#         mask = torch.tensor(mask_list)
#         # print(f'Inside self.mask, randomly generated mask_list = {mask_list}')
#         masked_y = y.clone()
#         masked_y[:, mask==1] = mask_idx
#         xo = torch.cat([x, masked_y], dim=1)
#         xu = y[:, mask==1] #order maintaineds
#         # print(f'test masked remains ({y[:, mask==1].shape}):' \
#         #     f'\n{masked_y[:, mask==1]} \nshould be of Size(batch_size, {num_masks}) with value {mask_idx}')###########
        
#         return xo, xu, u
        
        
#     def energy(self, idx:int, val: bool, rest_idx: torch.Tensor, latent: torch.Tensor, device='cpu') \
#         -> torch.Tensor: #目前唯一用torch.gather()的地方? 注意rest_idx需要提前unsqueeze(-1)变2D #cuda
#         '''
#         E(x_ui (=val) ; rest_idx)
        
#         Given a specified dimension, an index table and a latent value source, 
#         calculate the energy value for the specified index value on that dimension, 
#         or all the energy values along that dimension.
        
#         params:
#             idx: the specified index "i" to predict
#             val: whether the predict value "x_{u'_i}" is given, determines what to return
#             rest_idx: the conditional values (include pos i!), represented by a 3D tensor of Size(batch_size, |x_{u'_{<=i}}|, 1)
#             latent: the value source for energy calculation, represented by a 3D tensor of Size(batch_size, |x_{u'_{<=i}}|, num_classes) 
            
#         return:
#                 energy: Size(batch_size), if specified val;
#                 energy vector: Size(batch_size, num_classes), if val is not specified.  
#         '''
#         assert latent.dim() == rest_idx.dim() and latent.size(1) == rest_idx.size(1), \
#             f"latent.shape = {latent.shape}, rest_idx.shape = {rest_idx.shape}" #3, Size(batch_size, |u'i|, num_classes), Size(batch_size, |u'i|, 1)
#         if val:
#             energy = torch.gather(input=latent, dim=-1, index=rest_idx)
#             # except:
#             #     raise ValueError(f"gather energy from input(latent) of {latent.size()}" \
#             #         f"with index(rest_idx) of {rest_idx.size()} is not defined!")
#         else:
#             # expand the last dim of the original rest_idx from Size(batch_size, |ui'|, 1) to Size(batch_size, |ui'|, num_classes), where
#             # the original class value on ui position is replaced by arange(num_classes), 
#             # the class values on positions are copied for 'num_classes' of times
#             expanded_idx = torch.zeros(rest_idx.size(0), rest_idx.size(1), self.num_classes)
#             for pos in range(expanded_idx.size(1)):
#                 if pos == idx: #enumerate all vals on ui
#                     expanded_idx[:, pos, :] = torch.arange(self.num_classes) 
#                 else: #fill the original val on other positions for each class
#                     expanded_idx[:, pos, :] = rest_idx[:, pos, :].expand(-1, self.num_classes) 
#             expanded_idx = expanded_idx.to(torch.long).to(device)
#             # print(f'\nbefore expand, rest_idx({rest_idx.shape}): \n{rest_idx}, \nafter expand, expanded_idx({expanded_idx.shape}: \n{expanded_idx})')
#             energy = torch.gather(input=latent, dim=-1, index=expanded_idx)
#             # except:
#             #     raise ValueError(f"gather energy from input(latent) of {latent.size()}" \
#             #         f"with index(rest_idx) of {expanded_idx.size()} is not defined!")
                
#         return torch.sum(energy, dim=1) #sum along all ui positions

    
#     def sampling(self, sample_batch, sampler='gibbs', device='cpu', sampling_config=None, visual_ebms=None): #cuda
#         '''
#         Sampling on  the given sample batch, returns the prediction batch, the evaluation result and a viualization.

#         params:
#             - sample_batch (Size(batch_size, inp_len+out_len)): one-hot data batch (x, y)
#             - sampler (str): the sampling method, default: Gibbs
#             - sampling_config: batch_size, steps, temperature, metrics, etc.
#             - visual_ebms (VisualEBMs): an empty dic for storing EBM landscapes

#         returns:
#             - y_pred (Size(batch_size, out_len)): non-one-hot predicted sequence
#             - ebms_log (VisualEBMs.ebms_log): a filled dict containing the energy landscapes (num_vars x num_classes) and losses for a sample batch
#         '''
#         #___________configs_____________
#         batch_size = sampling_config.batch_size #should be one!!
#         steps_num = sampling_config.steps
#         T = sampling_config.temperature #?
#         time_step=visual_ebms.time_step #1
#         criterion = nn.CrossEntropyLoss()
#         softmax = nn.Softmax(dim=1)
#         #____________________________________
#         self.model.to(device)
#         sample_batch = sample_batch.to(device)
#         with torch.no_grad():
#             if sampler == 'gibbs':
#                 onehot_sample_batch = torch.nn.functional.one_hot(sample_batch.type(torch.long), self.num_classes) #one-hot transfer
#                 x = onehot_sample_batch[:, :self.inp_len, :]
#                 '''1. Randomly initialize 'y_pred' and sample an unmasking order 'pi'.'''
#                 y_pred = torch.randint(low=0, high=self.num_classes, size=(batch_size, self.out_len)).to(device) #not one-hot
#                 pi = torch.randperm(self.out_len) #unmasking order
#                 # print(f'\nInside sampling(), the randomly initialized y_pred({y_pred.shape}): \n{y_pred}' \
#                 #     f'\nunmasking order: {pi}')
#                 for k in range(1, self.ebm_num+1):
#                     # print(f'\n\n{k}-th EBM: ')
#                     yo = torch.unsqueeze(y_pred[:, pi[:k]], 2) #.to(device) #shuffled, Size(batch_size, |pi_k|, 1)
#                     '''2. Gibbs sampling on yo'''
#                     for t in range(steps_num): #TODO: log the energy landscapes and losses
#                         # print(f'\nInside {t}-th step: ')
#                         # y = sample_batch[:, self.inp_len:self.inp_len+k] #just for calculating the loss
#                         masked_y = torch.full((batch_size, self.out_len), 2).to(device)
#                         # print(f'\nmasked_y before: \n{masked_y}')
#                         masked_y[:, pi[:k]] = yo.squeeze(2) #这一步并不会改变yo.shape (Size(batch_size, k, 1))
#                         # print(f'\nmasked_y after: \n{masked_y}')
#                         model_input = torch.cat([sample_batch[:, :self.inp_len], masked_y], dim=1).float().to(device) #Size(batch_size, inp_len+out_len)
#                         gamma = self.model.forward(model_input, is_ebm=False) #Size(batch_size, out_len, num_classes)
#                         yo_energy = self.energy(idx=pi[0], val=True, rest_idx=yo, latent=gamma[:, pi[:k], :]) #Size(batch_size, 1)
#                         # print(f'\nyo_energy({yo_energy.shape}): \n{yo_energy}')
#                         #build conditional distribution (logits)
#                         energy_landscape = torch.zeros((k, self.num_classes), device=device) # first sample in each batch!
#                         losses = torch.zeros(k).to(device)
#                         yo_prime = yo.clone()
#                         for i in range(k):
#                             ei_dist = self.energy(idx=pi[i], val=False, rest_idx=yo, latent=gamma[:, pi[:k], :]) #Size(batch_size, num_classes)
#                             z_oi = torch.sum(torch.exp(-1*ei_dist), dim=-1) #Size(batch_size)
#                             expanded_zoi = z_oi.unsqueeze(1).expand_as(ei_dist) #Size(batch_size, num_classes)
#                             # Sample from the 1D conditional p(y_{o_i} | y_{o_-i})
#                             p_oi = torch.exp(-1*ei_dist) / expanded_zoi #Size(batch_size, num_classes)
                            
#                             try:
#                                 y_oi_prime = torch.multinomial(p_oi, num_samples=1) #Size(batch_size, 1)
#                             except:
#                                 raise RuntimeError(f'multinomial() input: p_oi seems contains nan.\n' \
#                                     f'p_oi({p_oi.shape}): \n{p_oi}\n' \
#                                     f'ei_dist({ei_dist.shape}): \n{ei_dist}\ngamma[:, pi[:k], :]({gamma[:, pi[:k], :].shape}): \n{gamma[:, pi[:k], :]}\n'\
#                                     f'gamma({gamma.shape}): \n{gamma}\n'\
#                                     f'model_input({model_input.shape}): \n{model_input}')
                            
#                             # print(f'\nThe sampled p_oi.shape = {p_oi.shape}') #Size(1, 2)
#                             yo_prime[:, i, :] = y_oi_prime
#                             if t % time_step == 0:
#                                 # Record the i-th row of the energy landscape and losses at (k, t)
#                                 energy_landscape[i, :] = ei_dist[0, :] #Size(1, num_classes)
#                                 yi = sample_batch[0, self.inp_len+pi[i]].view(-1).to(torch.long)
#                                 # print(f'\n CE between p_oi: {p_oi[0:1, :]} and yi: {yi}')
#                                 losses[i] = criterion(p_oi[0:1, :], yi)
#                         # end position iter
                        
#                         # try:
#                         yo_prime_energy = self.energy(idx=pi[0], val=True, rest_idx=yo_prime, latent=gamma[:, pi[:k], :]) #Size(batch_size, 1)
#                         # except:   
#                         #     raise RuntimeError(f'Error!: rest_idx(yo_prime).shape: {yo_prime.shape}, latent(gamma[:, pi[:k], :]).shape: {gamma[:, pi[:k], :].shape}')
                        
#                         '''3. Check if the energy decreases after a single Gibbs step'''
#                         mask = yo_prime_energy < yo_energy
#                         expanded_mask = mask.unsqueeze(2).expand_as(yo) #Size(batch_size, |pi_k|, 1)
#                         # print(f'\nyo_energy({yo_energy.shape}): \n{yo_energy}\nyo_prime_energy({yo_prime_energy.shape}): \n{yo_prime_energy}' \
#                         #     f'\nexpanded_mask({expanded_mask.shape}): \n{expanded_mask}')
#                         # print(f'\nBefore update, yo({yo.shape}): \n{yo}')
#                         yo[expanded_mask] = yo_prime[expanded_mask] #update the entire sample row of yo if the energy decreases
#                         # print(f'\nAfter update, yo({yo.shape}): \n{yo}')
#                         if t % time_step == 0: 
#                             # print(f'1st samples\' energy_landscape({energy_landscape.shape}): \n{energy_landscape}\nlosses({losses.shape}): {losses}')
#                             visual_ebms.screenshot(k, t+1, energy_landscape, torch.mean(losses.cpu()).item())
#                     #end of step iter
#                     y_pred[:, pi[:k]] = yo.squeeze(-1) 
#                     # print(f'predicted y = \n{y_pred}, ground_truth y = \n{sample_batch[:, self.inp_len:]} \nafter T = {steps_num} steps')
#                 #end of ebm iter
#                 if visual_ebms.visualize:
#                     return y_pred, visual_ebms.ebms_log
#                 else:
#                     return y_pred, None
#             else:
#                 raise NotImplementedError
    
#     '''Sample batch version'''
#     def train(self, data_batches: list[tuple[torch.Tensor, torch.Tensor]], train_config:dict, task_config:dict, visual_config:dict):
#         '''
#         Train a sequence of EBMs together.

#         params:
#             - data_batches (list(tensor.Size(batch_size, inp_len+out_len))): list of data_batches, each batch is a torch tensor.
#         '''
#         #____________hyper_params_____________
#         device= 'cpu' #'cuda'
#         # criterion = F.cross_entropy 
#         criterion = nn.CrossEntropyLoss().to(device)
#         corrupt_func = random_flip # Callable
#         optimizer = optim.Adam(self.model.parameters(), lr=train_config.lr)
#         epochs = train_config.epochs
#         batch_size = train_config.batch_size #100
#         sample_num = task_config.train_size
#         stat_path = f'./stats/train/{task_config.name}_{self.parameterization}.jsonl' #to store loss and ebms statistics
#         ckpts_path =  f'./ebm_ckpts/{task_config.name}_{self.parameterization}.pth' #to store the model checkpoints
#         early_stop_threshold = train_config.early_stop_threshold #0.5 (?)
#         if train_config.wandb:
#             wandb.login()
#             run = wandb.init(
#                 project=f'EBM_train-{task_config.name}_{self.parameterization}',  # Specify your project
#                 config={                        # Track hyperparameters and metadata
#                     "learning_rate": train_config.lr,
#                     "epochs": epochs,
#                 },
#             )
#         #_____________________________________
#         self.model.to(device)
#         train_start = time.time()
#         early_stop = False
#         with open(stat_path, 'w') as statfile:
#             print(f'Write to stat file at "{stat_path}" to store loss and EBMs statistics...')
#             for epoch in tqdm(range(epochs), desc='epoch'):
#                 for bi, batch in enumerate(data_batches):
#                     # visual_ebms = VisualizeEBMs(bi, task_config, visual_config) #sampling_config = None (steps=1)
#                     x, y = batch[:, :self.inp_len], batch[:, self.inp_len:]
#                     # print(f'{bi}-th sample batch: \nx({x.shape}): {x}\ny({y.shape}): {y}')
#                     '''1. Random mask and corrupt the sample'''
#                     xo, xu, u = self.mask(x, y, mask_idx=self.num_classes) #xo = x + y_masked, xu = masked y elements
#                     # print(f'\nxu: \n{xu}, \nu: \n{u}')
#                     neg_xu = corrupt_func(samples=xu, flip_range=self.num_classes) 
#                     # print(f'\n neg_xu: {neg_xu}')
#                     '''2. Generating latent vectors for energy calculation'''
#                     xo = xo.float().to(device)
#                     xu = xu.to(torch.long).to(device) #needed for indexing
#                     gamma = self.model.forward(xo, is_ebm=False) #Size(batch_size, out_len, num_classes)
#                     '''3. Pseudo-likelihood training for i = 1 to |u'| EBMs'''
#                     # logp_xu, l_contrast = torch.zeros((batch_size, len(u), self.num_classes), requires_grad=True, device=device), \
#                     logp_xu, l_contrast = [], \
#                         torch.zeros(batch_size, requires_grad=True, device=device) #Initialize
                        
#                     # energy_landscape = torch.zeros((len(u), self.num_classes), device=device) # first sample in each batch!
                    
#                     for i in range(len(u)): #iter through |u'| number of EBMs (one single backprop path per iter)
#                         i_order = torch.argsort(torch.tensor(u[:i+1]), dim=-1).tolist() #index within the u_{<=i} list
#                         # print(f'\n\ni_order: {i_order}, ui = {u[:i+1]}')
#                         condition_vals = torch.unsqueeze(xu[:, i_order], 2) # inclusive, the condition: x_{u'_{<=i}} # Size(bacth_size, |u'i|, 1)
#                         # calculate the logp_xu increment (no order issues)
#                         # print(f'Input data to the energy function: \nidx: {i}, \nrest_idx: \n{condition_vals}, \nlatent: \n{gamma[:, u[:i+1], :]}')
#                         # Note that rest_idx and gamma are already shuffled and of |u'i| length
#                         latent = gamma[:, u[:i+1], :]
#                         if latent.dim() == 2: #usually when |u'i| = 1
#                             latent = torch.unsqueeze(latent, 1)
#                         pos_ei_dist = self.energy(idx=i, val=False, rest_idx=condition_vals, latent=latent)  #Size(batch_size, num_classes)
#                         pos_ei = self.energy(idx=i, val=True, rest_idx=condition_vals, latent=latent)
#                         z_ui = torch.sum(torch.exp(-1*pos_ei_dist), dim=-1) #Size(batch_size)
#                         expanded_z_ui = z_ui.unsqueeze(1).expand_as(pos_ei_dist) #copy each row element by num_classes
#                         logp_xui = torch.log(torch.exp(-1*pos_ei_dist) / expanded_z_ui)
#                         # print(f'The {i}-th position, logp_xui({logp_xui.shape}) = \n{logp_xui}')
#                         # logp_xu = logp_xu + torch.log(torch.exp(-1*pos_ei_dist) / expanded_z_ui) #Eq.(0), Size(batch_size, num_classes)
#                         logp_xu.append(logp_xui.unsqueeze(1)) #Eq.(0), with sum replaced by concatenation
#                         # energy_landscape[i, :] = pos_ei_dist[0, :]
#                         # print(f'pos_ei_dist({pos_ei_dist.shape}): \n{pos_ei_dist}\nz_ui({z_ui.shape}): \n{z_ui}\nexpanded_z_ui(expanded_z_ui.shape): \n{expanded_z_ui}\nlogp_xu({logp_xu.shape}): \n{logp_xu}')
#                         # calculate the contrast loss increment
#                         neg_xu = torch.tensor(neg_xu, dtype=torch.long).to(device)
#                         condition_vals = torch.unsqueeze(neg_xu[:, i_order], 2)
#                         neg_ei = self.energy(idx=i, val=True, rest_idx=condition_vals, latent=latent) 
#                         pos_ei, neg_ei = torch.squeeze(pos_ei), torch.squeeze(neg_ei)
#                         # print(f'pos_ei({pos_ei.shape}): \n{pos_ei}\nneg_ei({neg_ei.shape}): \n{neg_ei}')
#                         l_contrast = l_contrast - torch.log(torch.exp(-1*pos_ei) / (torch.exp(-1*pos_ei) + torch.exp(-1*neg_ei))) #Eq.(2), batch_size
#                         # print(f'l_contrast: \n{l_contrast}')
#                         # break ####################
#                     #end of EBM iter
#                     '''4. Calculate loss and backprop'''
#                     logp_xu = torch.cat(logp_xu, dim=1)
#                     p_xu = torch.exp(logp_xu)
#                     # print(f'\n\nlogp_xu ({logp_xu.shape}): \n{logp_xu}')
#                     # print(f'\n\np_xu ({p_xu.shape}): \n{p_xu}')
                    
#                     logp_xu_flattened = logp_xu.view(-1, logp_xu.size(-1))
#                     xu_flattened = xu.view(-1) # Size(batch_size, |u'|) -> Size(batch_size*|u'|)
#                     l_ce = criterion(logp_xu_flattened, xu_flattened)
#                     l_contrast = l_contrast.mean()
#                     # print(f'l_ce: {l_ce}')
#                     # print(f"l_contrast: {l_contrast}")
                
#                     loss = l_ce + l_contrast ##scalar (batch-wise)
#                     # print(f'\nloss = {loss}')
#                     wandb.log({"l_ce": l_ce, "l_contrast": l_contrast, "loss": loss})
#                     loss.backward()
#                     optimizer.step()
                    
#                     if loss < early_stop_threshold:
#                         early_stop = True
#                         break
                    
#                     stat = {
#                         'epoch': epoch,
#                         'batch_id': bi,
#                         'l_ce': l_ce.cpu().item(),
#                         'l_contrast': l_contrast.cpu().item(),
#                         'loss': loss.cpu().item(),
#                         # 'ebms': visual_ebms,
#                     }
#                     statfile.write(json.dumps(stat)+'\n')
                        
#                     # if bi == 10: #####################test
#                     #     break#####################test
#                 #end of batch iter
#                 if early_stop:
#                     break
#                 # break#############test
#             #end of epoch iter
#             state_dict = self.model.state_dict()
#             torch.save(state_dict, ckpts_path)
#             print(f'\nFinished training, Checkpoints saved to {ckpts_path}')
#         #end of statfile write
#         spent = convert_time(train_start)
#         print(f'Training spent {spent[0]}:{spent[1]}:{spent[2]}.\n')
        
#     '''Single data sample version'''
#     def train_single(self, data, train_config, task_config):
#         for sample_id, (x, y) in enumerate(data):
#             print(f'{sample_id}-th sample: \nx({x.shape}): {x}\ny({y.shape}): {y}')
#             '''1. Random masking and corrupting the unobservables'''
#             u = self.random_mask(y.size(0)) 
#             xu, xo = self.mask(x, y, u) #TODO: concatenate masked y and x as the model input: x_o, the golden label is x_u
#             neg_xu, neg_xo = corrupt_func(xu, xo) #TODO: get corrupted xu and xo
#             '''2. Generating latent vectors for energy calculation'''
#             gamma = self.model.forward(xo, is_ebm=False) #return the latent vectors for each u_i given x_o
#             '''3. Pseudo-likelihood training for i = 1 to |u'| EBMs'''
#             u_prime = shuffle(u) # a list of shuffled unobserved indices
#             logp_xu, l_contrast = torch.zeros(self.num_classes, requires_grad=True), torch.tensor(0.0, requires_grad=True) #Initialize
#             for i, ui in enumerate(u_prime): #iter through |u'| number of EBMs (one single backprop path per iter)
#                 #TODO: 通过torch.gather和其他torch function实现求energy和q_ui
#                 # calculate the logp_xu increment
#                 pos_ei_dist = self.energy(dim=ui, val=None, rest_idx=xu, latent=gamma) #calculate the energies for all values along the given dim, #vector
#                 z_ui = torch.sum(torch.exp(-1*pos_ei_dist)) #scalar
#                 logp_xu += torch.log(torch.exp(-1*pos_ei_dist) / z_ui) #Eq.(0), vector
#                 # calculate the contrast loss increment
#                 pos_ei = pos_ei_dist[xu[i]]
#                 neg_ei = self.energy(dim=ui, val=neg_xu[i], rest_idx=neg_xu, latent=gamma)
#                 l_contrast -= torch.log(torch.exp(-1*pos_ei) / (torch.exp(-1*pos_ei) + torch.exp(-1*neg_ei))) #Eq.(2), scalar
#             #end of EBM iter
#             '''4. Calculate loss and backprop'''
#             l_ce = criterion(logp_xu, xu) #scalar
#             loss = l_ce + l_contrast
#             loss.backward()
#             optimizer.step()
                
#             break#####################test
#         #end of sample iter
            
#     def apply_metrics(self, pred, y, stat_dict, task_config=None):
#         '''
#         Apply a list of names of evaluation metrics to a pair of (prediction, y), return in dict

#         params:
#             - pred (Size(batch_size, out_len)): prediciton sequence
#             - y (Size(batch_size, out_len)): groundtruth
#             - stat_dict (dict): the final statistics dict on the evaluation set
#             - task_config: name, etc

#         returns:
#             - batch_stat (dict): batch statistics
#         '''
#         assert pred.shape == y.shape
#         batch_size = pred.size(0)
#         truth_matrix = (pred == y)
#         batch_dict = {}
#         for name in self.metrics:
#             if name == 'acc': #here we only return the counts
#                 batch_dict[name] = {}
#                 acc_count = torch.sum(truth_matrix, dim=1) #sum along the second dimension, Size(batch_size)
                
#                 he_count = torch.sum(acc_count == self.out_len).item() #number of exactly mathcing sequences
#                 se_count = torch.sum(acc_count).item() #number of exactly matching variables
#                 stat_dict['acc']['he'] += he_count
#                 stat_dict['acc']['se'] += se_count
#                 batch_dict['acc']['he'] = he_count / batch_size
#                 batch_dict['acc']['se'] = se_count / (batch_size * self.out_len)
#             else:
#                 raise NotImplementedError
#         batch_dict['pred'] = pred.cpu().tolist()

#         return batch_dict
    
#     def evaluate(self, val_data, store_stat=False, sampling_config=None, visual_config=None):
#         '''
#         Evaluate on the validation set.

#         params:
#             - val_data (list(tensor.Size(batch_size, inp_len+out_len)))): list of x+y batches, each batch is a 2D tensor
#             - store_stat (bool): whether store the evaluation statistics locally
#             - sampling_config
#             - visual_config
#         '''
#         #___________hyper params_____________
#         path_prefix = f'./stats/evaluate/{self.task_config.name}_{self.parameterization}'
#         visual_path = f'{path_prefix}_visual.jsonl'
#         stat_path = f'{path_prefix}_stat.jsonl'
        
#         task_config = {
#             'num_ebms': self.ebm_num,
#             'out_len': self.out_len,
#             'num_classes': self.num_classes,
#         }
#         stat_dict = {}
#         for name in self.metrics:
#             if name == 'acc':
#                 stat_dict[name] = {}
#                 stat_dict[name]['he'] = 0
#                 stat_dict[name]['se'] = 0
#             else:
#                 raise NotImplementedError
#         #____________________________________
#         with open(visual_path, 'w') as visualfile, open(stat_path, 'w') as statfile:
#             for bi, data in enumerate(tqdm(val_data)):
#                 # print(f'\nInside evaluate(), the {bi}-th data.shape: {data.shape}, data: \n{data}') #torch.Size([3, 22])
#                 visual_ebms = VisualizeEBMs(bi, task_config, visual_config, sampling_config)
#                 pred_batch, ebms_log = self.sampling(data, sampling_config=sampling_config, visual_ebms=visual_ebms)
#                 # if store_stat:
#                 y_batch = data[:, self.inp_len:].to(self.device) #Size(batch_size, out_len)
#                 batch_stat = self.apply_metrics(pred_batch, y_batch, stat_dict)
#                 batch_stat['x,y'] = data.cpu().tolist()
#                 statfile.write(json.dumps(batch_stat)+'\n')
#                 if ebms_log:
#                     visualfile.write(json.dumps(ebms_log)+'\n')
#                 # if bi == 10: ############################
#                 #     break############################
#                 # break ###################
            
#             #end of batch iter
#             # if store_stat:
#             for name in self.metrics:
#                 if name == 'acc':
#                     stat_dict[name]['he'] /= self.task_config.val_size
#                     stat_dict[name]['se'] /= (self.task_config.val_size * self.out_len)
#                 else:
#                     raise NotImplementedError
#             print(f'\nFinal Result:\n{stat_dict}')
#             if store_stat:
#                 statfile.write('\nFinal Result:\n' + json.dumps(stat_dict))
#         #end of stat write
#         print(f'\nFinished evaluation. \nstatistics saved to {stat_path}, \nvisualfile saved to: {visual_path}.')
        
class BERTSequentialEBMs():
    def __init__(self, tokenizer, task_config, d_model, n_layers, heads=8, device='cpu'):
        self.param_type = 'bert'
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.task_name = task_config.name
        if task_config.name.startswith('binary'):
            self.inp_len = task_config.inp_len
            self.out_len = task_config.out_len
            self.max_len = self.inp_len + self.out_len #+ 3 #the max length model can take in
        else:
            self.max_len = task_config.max_len #countdown: 50
        self.vocab_size = task_config.num_classes
        self.special_token_ids = {
            self.tokenizer.pad_token_id, 
            self.tokenizer.sep_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.mask_token_id,
            self.tokenizer.unk_token_id, 
        }
        self.special_tok_size = len(self.special_token_ids)
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.device = device ###########for debugging
        self.criterion = nn.CrossEntropyLoss()
        self.pe = PositionalEmbedding(self.vocab_size+1, max_len=self.max_len) #torch.Size([1, 50, 384])
        
        self._build_model()
        
    def _build_model(self): #initialize a bert
        self.model = DiscreteDiffusion(
            vocab_size=self.vocab_size, 
            max_len=self.max_len,
            hidden_size=self.d_model,
            n_layers=self.n_layers,
            heads=self.heads,
        ) #Assume inp_len >= out_len
        self.model.to(self.device)
    
    # wrapper
    def forward(self, bert_input, segment_label, is_ebm):
        if is_ebm:
            return self.model.forward(bert_input, segment_label, is_ebm)
        else:
            return self.model.forward(bert_input, segment_label, is_ebm), 0.0
        
    def energy(self, idx:int, val: bool, rest_idx: torch.Tensor, latent: torch.Tensor, \
        pe=True, batchalize=False, pos_ids=None) -> torch.Tensor:
        '''
        E(x_ui (=val) ; rest_idx)
        
        Given a specified dimension, an index table and a latent value source, 
        calculate the energy value for the specified index value on that dimension, 
        or all the energy values along that dimension.
        
        params:
            idx: the specified index "i" to predict
            val: whether the predict value "x_{u'_i}" is given, determines what to return
            rest_idx: the conditional values (include pos i!), represented by a 2D tensor of Size(|x_{u'_{<=i}}|, 1)
            latent: the value source for energy calculation, represented by a 2D tensor of Size(|x_{u'_{<=i}}|, num_classes) 
            pe: whether add positional embedding to the latent before calculating the energy
        return:
                energy: Size(1), if specified val;
                energy vector: Size(vocab_size), otherwise
        '''
        assert latent.dim() == rest_idx.dim() and latent.size(-2) == rest_idx.size(-2), \
            f"latent.shape = {latent.shape}, rest_idx.shape = {rest_idx.shape}"
        if val:
            energy = torch.gather(input=latent, dim=-1, index=rest_idx)
        else: #energy dist
            if batchalize:
                expanded_idx = torch.zeros(rest_idx.size(0), rest_idx.size(1), self.vocab_size)
                for pos in range(expanded_idx.size(1)):
                    if pos == idx: #enumerate all vals on ui
                        expanded_idx[:, pos, :] = torch.arange(self.vocab_size) 
                    else: #fill the original val on other positions for each class
                        expanded_idx[:, pos, :] = rest_idx[:, pos, :].expand(-1, self.vocab_size) 
            else:   
                expanded_idx = torch.zeros(rest_idx.size(0), self.vocab_size)
                for pos in range(expanded_idx.size(0)): #|u'i|
                    if pos == idx: #enumerate all vals on ui
                        expanded_idx[pos, :] = torch.arange(self.vocab_size) 
                    else: #fill the original val on other positions for each class
                        expanded_idx[pos, :] = rest_idx[pos, :].expand(self.vocab_size) 
                # print(f'\nrest_idx: \n{rest_idx}, \nexpanded_idx: \n{expanded_idx}')
            expanded_idx = expanded_idx.to(torch.long).to(self.device)
            energy = torch.gather(input=latent, dim=-1, index=expanded_idx) #Size(out_len, num_classes)
              
        if pe:
            # print(f'\nInside energy(), before sum, energy with val({val}).shape: {energy.shape}\n') #torch.Size([14, 31])
            # 考察下inference需不需要改 
            pos_emd = self.pe(energy).squeeze(0).to(self.device)
            # print(f'pos_emd.shape: {pos_emd.shape}')
            # print(f'latent.device: {latent.device}, pos_emd.device: {pos_emd.device}, pos_ids.device: {pos_ids.device}')
            pe_oui = pos_emd[pos_ids, :-1]
            # print(f'pe_oui.shape: {pe_oui.shape}') 
            if val:
                pe_oui = torch.gather(pe_oui, dim=-1, index=rest_idx)
            assert energy.shape == pe_oui.shape, \
                f'energy.shape({energy.shape}) does not match positional embedding\'s shape({pe_oui.shape})'
            energy = energy + pe_oui
        return -1 * torch.sum(energy, dim=-2) #sum along all ui positions #-1 *
        
    def gibbs_dist(self, energy_dist: torch.Tensor, energy_clip=True):
        '''
        Given the energy distribution at position i across all classes,
        calculate the Boltzmann (Gibbs) distribution. 
        
        params:
            energy_dist: Size(batch_size, num_classes)
            energy_clip: whether subtract the maximum energy before calculating the p_i distribution (to waive NaN), default to True
        return:
            p_i: Size(batch_size, num_classes)
        '''
        if energy_clip: 
            e_max = energy_dist.max()
            energy_dist = energy_dist - e_max
        
        z_i = torch.sum(torch.exp(-1*energy_dist), dim=-1) #Size(batch_size)
        expanded_zi = z_i.unsqueeze(1).expand_as(energy_dist) #Size(batch_size, num_classes)
        # Sample from the 1D conditional p(y_{o_i} | y_{o_-i})
        p_i = torch.exp(-1*energy_dist) / expanded_zi #Size(batch_size, num_classes)
        
        return p_i
    
    def draw_landscape(self, sample_id, k, t, full_latent, full_val, out_idx, groundtruth):
        '''
        At each time step, calculate and visualize the energy landscape with paths of inference and groundtruth.
        params:
        - sample_id
        - k: k-th EBM
        - t: t-th time step
        - full_latent: Size(full_len, vocab_size)
        - full_val: Size(full_len), previous pred
        - out_idx: Size(|u_<i|)
        - groundtruth: Size(full_len)
        '''
        
        '''
        partial_pred: 
        tensor([ 7,  8, 20,  7, 11, 20,  7, 10, 20,  7,  7,  1,  2,  2,  2,  2,  2,  2,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        '''
        # print(f'full_val: \n{full_val},\nsep_token_id: {self.tokenizer.sep_token_id}, eos_token_id: {self.tokenizer.eos_token_id}')
        sep_id = (full_val == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0].item()
        eos_id = (full_val == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0].item()
        # print(f'full output range: {sep_id+1}:{eos_id}')
        # print(f'full_val: \n{full_val}')
        # 1. calculate the energy landscape (global landscape with full output length)
        full_len = full_val.size(0)
        # partial_out_len = out_idx.size(0) #output only
        # inp_len = full_idx.size(0) - partial_out_len
        # full_out_len = full_len - inp_len
        landscape = []
        argmin = [] #the greedy pred from the energy dist
        for i in range(sep_id+1, eos_id): #full output len
            ei_dist = self.energy(
                idx=i, 
                val=False, 
                rest_idx=full_val.view(-1, 1), 
                latent=full_latent, 
                batchalize=False, 
                pos_ids=torch.arange(full_len)
            )
            landscape.append(ei_dist.unsqueeze(0))
            if torch.any(torch.eq(out_idx, i)):
                # print(f'find token to be unmasked at i={i}')
                p_oi = self.gibbs_dist(ei_dist.unsqueeze(0), energy_clip=True)
                y_oi_prime = torch.argmax(p_oi[:,self.special_tok_size:]) + self.special_tok_size #max (trivial case)
                argmin.append(int(y_oi_prime.squeeze()))
        landscape = torch.cat(landscape, dim=0) #Size(tok_len, vocab_size) 需要transpose一下以便显示
        
        # 2. visaulize the energy ladnscape
        tok_num, vocab_size = landscape.size(0), landscape.size(1)
        landscape = landscape.detach().cpu().numpy()
        pred_route = out_idx.detach().cpu().numpy()#.astype(int)
        gold_route = groundtruth[sep_id+1:eos_id].detach().cpu().numpy()#.astype(int)
        argmin = np.array(argmin)
        # print(f'\ntok_num: {tok_num}, vocab_size: {vocab_size}') #16, 31
        # print(f'landscape({landscape.shape}): \n{landscape}, \npred_route: {pred_route}, \ngold_route: {gold_route}, \nargmin: {argmin}')
        # print(f'out_idx: {out_idx}')
        # raise
        X, Y = np.meshgrid(np.arange(tok_num), np.arange(vocab_size)) #X: (31,16), Y: (31,16)
        fig = go.Figure(data=[go.Surface(z=landscape.T, x=X, y=Y, colorscale='Hot', \
            opacity=0.5)]) #other colorscale: Viridis #show route: 0.2; show landscape: 0.5
        
        
        # print(f'pred_route: {pred_route}, gold_route: {gold_route}, argmin: {argmin}')
        curve_x = (out_idx-(sep_id+1)).cpu().detach().numpy()
        # print(f'curve_x: {curve_x}')
        if len(pred_route) == 1:
            # Plot the sampled marker (marker1)
            curve1_y = pred_route
            curve1_z = landscape[np.arange(out_idx.size(0)), pred_route]
            fig.add_trace(go.Scatter3d(
                x=curve_x, y=curve1_y, z=curve1_z,
                mode='markers',  # Use 'markers' to plot a single point
                marker=dict(color='blue', size=6),  # Customize the marker
                name='sampled'
            ))
            # Plot the argmin marker (marker2)
            curve2_y = argmin
            curve2_z = landscape[np.arange(out_idx.size(0)), argmin]
            # print(f'curve2_x: {curve_x}, curve2_y: {curve2_y}, curve2_z: {curve2_z}')
            fig.add_trace(go.Scatter3d(
                x=curve_x, y=curve2_y, z=curve2_z,
                mode='markers',
                marker=dict(color='red', size=6),
                name='argmin energy'
            ))
        else:
            # Plot the sampled curve (curve1)
            curve1_y = pred_route
            curve1_z = landscape[curve_x, pred_route]
            fig.add_trace(go.Scatter3d(
                x=curve_x, y=curve1_y, z=curve1_z,
                mode='lines',
                line=dict(color='blue', width=2),
                name='sampled'
            ))
            # Plot the argmin curve (curve2)
            curve2_y = argmin
            curve2_z = landscape[curve_x, argmin]
            fig.add_trace(go.Scatter3d(
                x=curve_x, y=curve2_y, z=curve2_z,
                mode='lines',
                line=dict(color='red', width=5),
                name='argmin energy'
            ))
        # Plot the groundtruth curve (curve3)
        curve3_x = np.arange(eos_id-sep_id-1)
        curve3_y = gold_route
        curve3_z = landscape[np.arange(eos_id-sep_id-1), gold_route]
        # print(f'curve3_x: {curve3_x}, curve3_y: {curve3_y}, curve3_z: {curve3_z}')
        fig.add_trace(go.Scatter3d(
            x=curve3_x, y=curve3_y, z=curve3_z,
            mode='lines',
            line=dict(color='green', width=6),
            name='groundtruth'
        ))
        
        # Set labels and title
        title = f'Sample{sample_id}\'s Landscape at k={k}, t={t}' 
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, tok_num-1], title='Token Position'),
                yaxis=dict(range=[0, vocab_size-1], title='Token ID'),
                zaxis_title='Energy',
            ),
            title=title,
            legend=dict(
                orientation="h",  # Horizontal orientation
                y=-0.2,  # Position below the plot
                x=0.5,  # Centered horizontally
                xanchor="center",
                yanchor="top"
            ),
            coloraxis_colorbar=dict(
                len=0.5,  # Length of the color bar
                y=-0.1,  # Position below the plot
                yanchor="bottom",
                x=0.5,  # Centered horizontally
                xanchor="center"
            )
        )
        
        fig.show()
        
        # raise
        # # # visuzlize in a Dash HTML app (view in webpage)
        # # app = Dash()
        # # app.layout = html.Div([
        # #     dcc.Graph(figure=fig)
        # # ])
        # # # app.run(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
        # # host_name = '10.64.199.251' #ssh_client env var
        # # port = 8848
        # # app.run(host=host_name, port=port, debug=True)
        # # print(f'Finished... Mayber you can open this webpage: \nhttp://{host_name}:{port}')
        # # raise
        return fig

    '''Simplified version of sampling'''
    def sampling_new(self, order_label, partial_pred, sample_batch, sampler='gibbs', \
        sampling_times=10, visualize=False, batch_id=None):
        batch_size = sample_batch['input'].size(0)
        pred = []
        landscapes = {}
        if sampler == 'gibbs':
            for b in range(batch_size):
                yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                    (sample_batch['schedule_label'][b] <= order_label)).\
                    nonzero(as_tuple=True)[0].to(self.device)
                # print(f'yo_idx: {yo_idx}')
                # print(f'partial_pred: {partial_pred}')
                full_val = partial_pred[b]
                yo = full_val[yo_idx]
                sep_id = (full_val == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0].item()
                gamma = self.model.forward(full_val.unsqueeze(0), None, is_ebm=True)\
                    .view(-1, self.vocab_size)
                for t in range(sampling_times):
                    if b == 0 and visualize and t == sampling_times-1:
                        landscape = self.draw_landscape(
                            sample_id=batch_id*batch_size,
                            k=order_label,
                            t=t,
                            full_latent=gamma,
                            full_val=full_val.unsqueeze(-1),
                            out_idx=yo_idx,
                            groundtruth=sample_batch['label'][b], #full_length label
                        )
                        landscapes[t] = landscape
                    
                    yo_energy = self.energy(
                        idx=sep_id+0,
                        val=True,
                        rest_idx=full_val.unsqueeze(-1),
                        latent=gamma,
                        batchalize=False,
                        pos_ids=torch.arange(full_val.size(0))
                    )
                    
                    full_val_cand, yo_cand = full_val.clone(), yo.clone()
                    for i in range(yo_idx.size(0)):
                        ei_dist = self.energy(
                            idx=yo_idx[i],
                            val=False,
                            rest_idx=full_val_cand.unsqueeze(-1),
                            latent=gamma,
                            pos_ids=torch.arange(full_val_cand.size(0))
                        )
                        p_oi = self.gibbs_dist(ei_dist.unsqueeze(0), energy_clip=True)
                        # y_oi_cand = torch.multinomial(p_oi[:,self.special_tok_size:], 1) + self.special_tok_size
                        y_oi_cand = torch.argmax(p_oi[:,self.special_tok_size:]) + self.special_tok_size
                        full_val_cand[yo_idx[i]] = y_oi_cand.squeeze()
                    yo_cand_energy = self.energy(
                        idx=sep_id+0,
                        val=True,
                        rest_idx=full_val_cand.unsqueeze(-1),
                        latent=gamma,
                        batchalize=False,
                        pos_ids=torch.arange(full_val_cand.size(0))
                    )
                    if yo_cand_energy.item() < yo_energy.item():
                        full_val = full_val_cand
                # end of t iter
                pred.append(full_val.view(1, -1))
            # end of batch iter
            pred = torch.cat(pred, dim=0)
            stat = {}
            if visualize:
                stat['landscapes'] = landscapes     
        else:
            raise NotImplementedError   
    
        return pred, stat
        
    '''simplified(corrected?) non-batchalized version'''
    def sampling(self, order_label:int, partial_pred: torch.Tensor, sample_batch:dict, \
        sampler='gibbs', sampling_times=10, visualize=False, batch_id=None):
        '''
        Sampling on a partially masked sample batch. Gibbs sampling by default.
        
        params: 
            order_label: k (1-indexed)
            partial_pred: Size(batch_size, seq_len)
            sample_batch: dict
            samping_times: T
            
        return:
            updated_partial_pred: Size(batch_size, seq_len)
            sth: dict, contains loss_list, energy_list, visual_ebms.log, or testing infos 
        '''
        # 1. Initialize yo and inputs to the model.forward and energy calcualtion
        batch_size = sample_batch['input'].size(0)
        losses = []
        energies = []
        landscapes = {}
        pred = []
        if sampler == 'gibbs':
            for b in range(batch_size):
                previous_pred = partial_pred[b]
                yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                    (sample_batch['schedule_label'][b] <= order_label)).nonzero(as_tuple=True)[0].to(self.device)
                # print(f'yo_idx.shape({yo_idx.shape}): \n{yo_idx}') #Size(|o|)
                # print(f'previous_pred.device: {previous_pred.device}, yo_idx.device: {yo_idx.device}, self.device: {self.device}')
                yo = previous_pred[yo_idx]
                # print(f'yo({yo.shape}): \n{yo}')
                model_input = sample_batch['input'][b].clone()
                # inp_idx = model_input.nonzero(as_tuple=True)[0].to(self.device)
                sep_id = (model_input == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0].item()              
                # inp = model_input[inp_idx].to(self.device)
                # inp_len = inp.size(0)
                
                # # Initialize Method1: fully random
                # model_input[model_input == 3] = 0 #replace the MASK with 0
                # model_input += previous_pred
                # Initialize Method2: keep the history prediction from previous k-1 iterations
                if order_label != 1:
                    history_yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                        (sample_batch['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
                    model_input[history_yo_idx] = previous_pred[history_yo_idx]
                
                # print(f"\norder_label={order_label}, bert_input: {sample_batch['input'][b]}\nprevious_pred: {previous_pred}"\
                #     f"\nmodel_input: {model_input}")
                
                '''ordered, non-zero length'''
                curr_state = model_input.to(self.device)
                curr_state[yo_idx] = yo
                # print(f'curr_state: \n{curr_state}, \nself.special_tok_size: {self.special_tok_size}')
                # print(f'yo_idx: {yo_idx}, yo: {yo}')
                inp_mask = (curr_state > self.special_tok_size)
                yo_mask = (sample_batch['schedule_label'][b] > 0) & (sample_batch['schedule_label'][b] <= order_label)
                full_idx_mask = (inp_mask | yo_mask) #whether tokens are observable (the initial yo tokens are [MASK] but observable)
                full_idx = (full_idx_mask).nonzero(as_tuple=True)[0]
                full_val = curr_state[full_idx]
                # print(f'full_idx: {full_idx}, full_val: {full_val}')
                
                
                gamma = self.model.forward(model_input.unsqueeze(0), \
                        None, is_ebm=True).view(-1, self.vocab_size) # sample_batch['segment_label'][b].unsqueeze(0)
                
                for t in range(sampling_times):
                    # print(f'\n____\nStart t={t}-th sampling...\n')
                    if b==0 and visualize and order_label==3 and t==0: #only calculate each first sample within a batch
                        # print(f"Inside sampling(), schedule_label: \n{sample_batch['schedule_label'][b]}, \ninput: \n{sample_batch['input'][b]}, \nyo_idx: {yo_idx}")
                        # print(f'self.special_tok_size: {self.special_tok_size}')
                        landscape = self.draw_landscape(
                            sample_id=batch_id*batch_size, #first sample in each batch
                            k=order_label,
                            t=t,
                            full_latent=gamma,  #full length
                            full_val=curr_state, #full length previous pred
                            out_idx=yo_idx,  
                            groundtruth=sample_batch['label'][b],
                        )
                        landscapes[t] = landscape
                    
                    yo_energy = self.energy(
                        idx=sep_id+1, #random...
                        val=True, 
                        rest_idx=full_val.unsqueeze(-1), 
                        latent=gamma[full_idx, :], 
                        batchalize=False, 
                        pos_ids=full_idx,
                    )
                    new_full_val = full_val.clone() #full_val和new_full_val是不断迭代T次的（略显繁琐
                    new_yo = yo.clone()
                    
                    # 2. gibbs sampling on each masked position (update on full_val!)
                    for i in range(yo_idx.size(0)): #iter |o|
                        removed_zeros = (new_full_val[:yo_idx[i]] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0] #the disgarded zeros in this partial input during nonzero()
                        removed_zeros = removed_zeros.numel()
                        
                        # sample on position i
                        # print(f'inputs to the energy: yo_prime: {yo_prime}, gamma[yo_idx,:]: {gamma[yo_idx, :]}')
                        ei_dist = self.energy(
                            idx=yo_idx[i]-removed_zeros, 
                            val=False, 
                            rest_idx=new_full_val.unsqueeze(-1), 
                            latent=gamma[full_idx, :], 
                            batchalize=False, 
                            pos_ids=full_idx
                        )
                        # print(f'ei_dist: \n{ei_dist}')
                        p_oi = self.gibbs_dist(ei_dist.unsqueeze(0), energy_clip=True) #Size(1,6) #has to clip (otherwise NaN)
                        
                        #_________forcing ignoring/considering the special tokens_______
                        # print(f'p_oi({p_oi.shape}): \n{p_oi},\n ei_dist({ei_dist.shape}): \n{ei_dist}') 
                        y_oi_prime = torch.multinomial(p_oi[:,self.special_tok_size:], 1) + self.special_tok_size #sample
                        # print(f'\ninside Gibbs sampling, p_oi: {p_oi}, self.special_tok_size: {self.special_tok_size}')
                        # raise
                        # y_oi_prime = torch.argmax(p_oi[:,self.special_tok_size:]) + self.special_tok_size #max (trivial case)
                        # y_oi_prime = torch.multinomial(p_oi, 1)
                        #___________________________________________________
                        
                        # update the sampled  yo'_i to previous_full_val (nonzero full vals)
                        # print(f'i: {i}, yo_idx[i]: {yo_idx[i]}')
                        # print(f'removed_zeros: {removed_zeros}')
                        new_full_val[i] = y_oi_prime.squeeze()
                        new_yo[i] = y_oi_prime.squeeze()
                        # print(f'i={i}: \n- y_oi\': {y_oi_prime.item()}, \n- ei_dist: {ei_dist}, \n- logits: {gamma[yo_idx[i], :]}')
                    # 3. update yo with yo' if the energy decreases
                    yo_prime_energy = self.energy(
                        idx=sep_id+1, #random... 
                        val=True, 
                        rest_idx=new_full_val.unsqueeze(-1), 
                        latent=gamma[full_idx, :], 
                        batchalize=False, 
                        pos_ids=full_idx
                    )
                    # print(f'yo\' energy: {yo_prime_energy}')
                    if yo_prime_energy.item() < yo_energy.item():
                        # yo = yo_prime
                        full_val = new_full_val
                        curr_state[full_idx] = full_val
                        # update model input as well
                        # print(f'yo\' energy is smaller, before update, previous_pred: {previous_pred}')
                        # previous_pred[yo_idx] = yo #?
                        previous_pred[full_idx] = full_val
                        # print(f'after update, previous_pred: {previous_pred}, \nbert_input: {sample_batch["bert_input"][b]}')
                        model_input = sample_batch['input'][b].clone()
                        model_input[previous_pred != 0] = previous_pred[previous_pred != 0]
                        # model_input[model_input == (self.special_tok_size-1)] = 0
                        # model_input += previous_pred
                        # print(f'model_input: {model_input}')
                    # 4. Record partial prediction, losses(using logits) and energies (last sample in the batch)
                    if b == 0:
                        loss = self.criterion(gamma[yo_idx, :], sample_batch['label'][b][yo_idx])
                        losses.append(round(loss.item(),2))
                        energies.append(round(yo_energy.item(),2))
                #end of t iter
                pred.append(previous_pred.view(1,-1))
                # break ####################test
            #end of inner-batch iter
        #end of 'gibbs'
        
        elif sampler == 'argmin_energy': 
            '''
            similar to 'sft', but in AR style
            (due to sequence of k, though history tokens might change) #greedy optimize
            '''
            for b in range(batch_size):
                previous_pred = partial_pred[b]
                yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                    (sample_batch['schedule_label'][b] <= order_label)).nonzero(as_tuple=True)[0] #<=
                model_input = sample_batch['input'][b].clone()
                # Initialize Method2: keep the history prediction from previous k-1 iterations
                if order_label != 1:
                    history_yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                        (sample_batch['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
                    model_input[history_yo_idx] = previous_pred[history_yo_idx]
                
                # forward pass with softmax in a single run (gamma.size = (30,6))
                gamma = self.model.forward(model_input.unsqueeze(0), \
                        None, is_ebm=False).view(-1, self.vocab_size) # sample_batch['segment_label'][b].unsqueeze(0)
                gamma[:, 0] = IGNORE_INDEX
                # print(f'gamma.shape: {gamma.shape}') #torch.Size([50, 31])
                yo = gamma[yo_idx, :].argmax(dim=-1) #same as argmin_energy #argmin for '_w_mask.pth'
                # print(f'b={b}, yo({yo.shape}): {yo}')
                previous_pred[yo_idx] = yo
                pred.append(previous_pred.unsqueeze(0)) #view(1,-1)
                # record loss and energy (first sample per batch)
                if b == 0:
                    # print(f'k={order_label}, gamma: \n{gamma}')
                    loss = self.criterion(gamma[yo_idx, :], sample_batch['label'][b][yo_idx])
                    yo_energy = self.energy(idx=0, val=True, rest_idx=yo.unsqueeze(-1), \
                        latent=gamma[yo_idx, :], batchalize=False, pos_ids=yo_idx)
                    losses.append(round(loss.item(),2))
                    energies.append(round(yo_energy.item(),2))
            # end of inner-batch iter
        elif sampler == 'argmax_logits':
            raise
        else:
            raise NotImplementedError(f"The sampler: {sampler} is not defined.")
        pred = torch.cat(pred, dim=0)
        # print(f'sampled pred[0] ({pred.shape}): {pred[0]}')
        
        stat = {'losses': losses, 'energies': energies}
        if visualize: 
            stat['landscapes'] = landscapes
        
        return pred, stat
    
       
    def calculate_contrast_loss(self, pos_latent, neg_latent, pos_label, neg_label, \
        pos_input, neg_input, form='softmax', threshold=2, alpha=0.2):
        # 1. Prepare all the computing factors needed
        us = []
        pis = []
        for _, (label, input) in enumerate(zip([pos_label, neg_label], \
            [pos_input, neg_input])):
            u = torch.nonzero(label != self.tokenizer.pad_token_id).squeeze() #Size(|u|)
            if u.dim():
                pi = torch.arange(u.size(0))
            else:
                pi = torch.tensor([0])
                u = torch.tensor([u])
            us.append(u), pis.append(pi)
        pos_eis, neg_eis = [], []
        pos_ei_dists, neg_ei_dists = [], []
        for i in range(len(us[1])): #TODO: 优化 complexity
            for is_neg, (u, pi, input, label, latent) in enumerate(zip(us, pis, \
                [pos_input, neg_input], [pos_label, neg_label], [pos_latent, neg_latent])):
                u, latent = u.to(self.device), latent.to(self.device)
                full_vals = input.clone()
                full_vals[u[:i+1]] = label[u[:i+1]]         
                ei_dist = self.energy(
                    idx=u[i], #o_len+pi[i], 
                    val=False, 
                    rest_idx=full_vals.view(-1, 1), 
                    latent=latent,
                    pos_ids=torch.arange(full_vals.size(0))
                ) #Size(1, num_classes)
                ei = ei_dist[full_vals[u[i]]].view(1, -1)
                if is_neg:
                    neg_eis.append(ei), neg_ei_dists.append(ei_dist) 
                else:
                    pos_eis.append(ei), pos_ei_dists.append(ei_dist)
        assert len(pos_eis) != 0 and len(neg_eis) != 0 and len(pos_eis) == len(neg_eis), f'u: {u}\npos_label({pos_label.shape}): {pos_label}, \nneg_label({neg_label.shape}): {neg_label}'
        # 2. Compute contrast losses of different functional forms
        if form == 'l2':
            contrast_criterion = nn.MSELoss(reduction='mean')
            pos_e, neg_e = torch.cat(pos_eis, dim=0), torch.cat(neg_eis, dim=0)
            assert pos_e.size(0) == neg_e.size(0) == len(us[1]), f'energy tensors\' length mismatch! pos_e.size: {pos_e.size()}, neg_e.size(): {neg_e.size()}, |u\'|: {len(us[1])}' #|u'|
            l2 = contrast_criterion(pos_e, neg_e)
            
            # pos_e_dist = torch.cat(pos_ei_dists, dim=0) #Size(|u'|, num_classes)
            # pos_e_dist = pos_e_dist - torch.min(pos_e_dist)
            # pos_reg = torch.mean(-1*pos_e_dist)
            # reg = torch.sum(alpha * (torch.pow(pos_e, 2) + torch.pow(neg_e, 2)))
            # print(f'pos_e.size: {pos_e.size()}, neg_e.size(): {neg_e.size()}, l2 contrast_loss: {contrast_loss.size()}')
            contrast_loss = torch.pow(torch.clamp(threshold - l2, min=0.0), 2) #+ pos_reg # + reg
        elif form == 'hinge':
            contrast_criterion = nn.MultiMarginLoss(margin=5, reduction='mean')
            pos_e_dist = torch.cat(pos_ei_dists, dim=0) #Size(|u'|, num_classes)
            neg_e_dist = torch.cat(neg_ei_dists, dim=0) #Size(|u'|)
            assert pos_e_dist.size() == neg_e_dist.size(), print(f'Mismatch!! pos_e_dist: {pos_e_dist.size()}, neg_e_dist: {neg_e_dist.size()}')
            # print(f'inputs to the hinge loss: pos_e_dist: {pos_e_dist.shape}, pos_e_dist: {pos_e_dist.shape}, neg_pred: {torch.argmax(neg_e_dist, dim=0)}')
            # take the mean of the two hinge losses after interchange the pos and neg tensors
            
            # contrast_loss = contrast_criterion(pos_e_dist, torch.argmax(neg_e_dist, dim=0))
            contrast_loss = (contrast_criterion(pos_e_dist, torch.argmax(neg_e_dist, dim=0)) + \
                contrast_criterion(neg_e_dist, torch.argmax(pos_e_dist, dim=0))) / 2
        elif form == 'reg':
            pos_e, neg_e = torch.cat(pos_eis, dim=0), torch.cat(neg_eis, dim=0)
            contrast_loss = torch.sum(alpha * (torch.pow(pos_e, 2) + torch.pow(neg_e, 2)))
            
        contrast_loss = (contrast_loss).squeeze()
        return contrast_loss
    
    def calculate_contrast_loss_old(self, pos_latent, neg_latent, pos_label, neg_label, \
        pos_input, neg_input, form='softmax', threshold=2, alpha=0.2):
        '''
        Non-batchalized, calcualte the contrast loss based on Eq.(2), take sum from all classes
        All tensor parameters: Size(max_len)
        
        functional forms: 
            - softmax (diverged): compute the logsoftmax of each pos-neg enegy pair at each position, them sum up
            - L2: compute the batch-averaged L2 loss between positive and negative energy tensors
            - hinge: compute the multiclass Hinge loss between the positive energy-based distribution and an argmax negative energy label
        '''
        raise
        # 1. Prepare all the computing factors needed
        os_ = []
        u_primes = []
        pis = []
        for _, (label, input) in enumerate(zip([pos_label, neg_label], \
            [pos_input, neg_input])):
            u = torch.nonzero(label != self.tokenizer.pad_token_id).squeeze() #Size(|u|)
            o_mask = (input != self.tokenizer.mask_token_id) & \
                (input != self.tokenizer.pad_token_id)
            o = torch.nonzero(o_mask, as_tuple=True)[0]
            o_len = o.size(0)
            if u.dim():
                # pi = torch.randperm(u.size(0)) #a random estimating order
                # u_prime = u[pi]
                pi = torch.arange(u.size(0))
                u_prime = u
            else: # u is a single value
                pi = torch.tensor([0])
                u, u_prime = torch.tensor([u]), torch.tensor([u])
            if o.dim() == 0:
                o = torch.tensor([o])
            os_.append(o), u_primes.append(u_prime), pis.append(pi)
        pos_eis, neg_eis = [], []
        pos_ei_dists, neg_ei_dists = [], []
        pos_label, neg_label, pos_input, neg_input = pos_label.unsqueeze(-1), \
            neg_label.unsqueeze(-1), pos_input.unsqueeze(-1), neg_input.unsqueeze(-1)
        for i in range(len(u_primes[1])): #TODO: 优化 complexity
            for is_neg, (o, u_prime, pi, input, label, latent) in enumerate(zip(os_, u_primes, pis, \
                [pos_input, neg_input], [pos_label, neg_label], [pos_latent, neg_latent])):
                # print(f'\ninput({input.shape}): \n{input}, \no({o.shape}): \n{o}')
                u_prime, o, latent = u_prime.to(self.device), o.to(self.device), latent.to(self.device)
                
                '''ordered non-zero o and u tokens'''
                full_vals = input.clone()
                full_vals[u_prime[:i+1], :] = label[u_prime[:i+1], :]
                inp_mask = (full_vals > self.special_tok_size)
                # yo_mask = torch.isin(torch.arange(full_vals.size(0)).to(self.device), u_prime[:i+1])
                yo_mask = (label.view(-1) > 0).to(self.device)
                full_idx_mask = (inp_mask | yo_mask) #whether tokens are observable (the initial yo tokens are [MASK] but observable)
                
                pos_ids = (full_idx_mask).nonzero(as_tuple=True)[0].squeeze(-1) #full_vals
                full_vals = full_vals[pos_ids, :]
                full_latent = latent[pos_ids, :]
                removed_zeros = (full_vals[:u_prime[i], :].squeeze(-1) == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0] #the disgarded zeros in this partial input during nonzero()
                removed_zeros = removed_zeros.numel()
                
                '''inordered concatenation'''
                # full_vals = torch.cat([input[o, :], label[u_prime[:i+1], :]], dim=0)
                # full_latent = torch.cat([latent[o, :], latent[u_prime[:i+1], :]], dim=0)
                # pos_ids = torch.cat([o.view(1, -1), u_prime[:i+1].view(1, -1)], dim=1).squeeze()
                
                
                ei = self.energy(
                    idx=u_prime[i]-removed_zeros, #o_len+pi[i], 
                    val=True, 
                    rest_idx=full_vals, 
                    latent=full_latent, 
                    pos_ids=pos_ids
                ) #Size(1)
                ei_dist = self.energy(
                    idx=u_prime[i]-removed_zeros, #o_len+pi[i], 
                    val=False, 
                    rest_idx=full_vals, 
                    latent=full_latent,
                    pos_ids=pos_ids
                ) #Size(1, num_classes)
                
                if is_neg:
                    neg_eis.append(ei)
                    neg_ei_dists.append(ei_dist) 
                else:
                    pos_eis.append(ei)
                    pos_ei_dists.append(ei_dist)
        assert len(pos_eis) != 0 and len(neg_eis) != 0, f'u_prime: {u_prime}\npos_label({pos_label.shape}): {pos_label}, \nneg_label({neg_label.shape}): {neg_label}'
        # 2. Compute contrast losses of different functional forms
        if form == 'softmax': #diverge error
            all_eis = torch.cat(pos_eis+neg_eis, dim=0) #Size(|u'i|)
            max_ei = all_eis.max()
            contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)
            for pos_ei, neg_ei in zip(pos_eis, neg_eis):
                pos_ei, neg_ei = pos_ei-max_ei, neg_ei-max_ei
                # sigmoid because the raw contrast loss is huge (6k+)
                contrast_loss = contrast_loss - ( \
                    torch.log(torch.exp(-1*pos_ei) / \
                        (torch.exp(-1*pos_ei) + torch.exp(-1*neg_ei))))
        elif form == 'l2':
            contrast_criterion = nn.MSELoss(reduction='mean')
            pos_e, neg_e = torch.cat(pos_eis, dim=0), torch.cat(neg_eis, dim=0)
            assert pos_e.size(0) == neg_e.size(0) == len(u_primes[1]), f'energy tensors\' length mismatch! pos_e.size: {pos_e.size()}, neg_e.size(): {neg_e.size()}, |u\'|: {len(u_primes[1])}' #|u'|
            l2 = contrast_criterion(pos_e, neg_e)
            reg = torch.sum(alpha * (torch.pow(pos_e, 2) + torch.pow(neg_e, 2)))
            # print(f'pos_e.size: {pos_e.size()}, neg_e.size(): {neg_e.size()}, l2 contrast_loss: {contrast_loss.size()}')
            contrast_loss = torch.pow(torch.clamp(threshold - l2, min=0.0), 2) # + reg
        elif form == 'hinge':
            contrast_criterion = nn.MultiMarginLoss(margin=5, reduction='mean')
            pos_e_dist = torch.cat(pos_ei_dists, dim=0) #Size(|u'|, num_classes)
            neg_e_dist = torch.cat(neg_ei_dists, dim=0) #Size(|u'|)
            assert pos_e_dist.size() == neg_e_dist.size(), print(f'Mismatch!! pos_e_dist: {pos_e_dist.size()}, neg_e_dist: {neg_e_dist.size()}')
            # print(f'inputs to the hinge loss: pos_e_dist: {pos_e_dist.shape}, pos_e_dist: {pos_e_dist.shape}, neg_pred: {torch.argmax(neg_e_dist, dim=0)}')
            # take the mean of the two hinge losses after interchange the pos and neg tensors
            
            # contrast_loss = contrast_criterion(pos_e_dist, torch.argmax(neg_e_dist, dim=0))
            contrast_loss = (contrast_criterion(pos_e_dist, torch.argmax(neg_e_dist, dim=0)) + \
                contrast_criterion(neg_e_dist, torch.argmax(pos_e_dist, dim=0))) / 2
        elif form == 'reg':
            pos_e, neg_e = torch.cat(pos_eis, dim=0), torch.cat(neg_eis, dim=0)
            contrast_loss = torch.sum(alpha * (torch.pow(pos_e, 2) + torch.pow(neg_e, 2)))
            
        contrast_loss = (contrast_loss).squeeze()
        return contrast_loss
    
    # simplified 
    def pseudolikelihood(self, latent, mlm_label, mlm_input):
        assert latent.size(0) == mlm_label.size(0) == mlm_input.size(0)
        u = torch.nonzero(mlm_label != self.tokenizer.pad_token_id).squeeze() #Size(|u|)
        if u.dim() == 0:
            u = torch.tensor([u])
        logp_xu = []
        if len(u) == 0:
            print(f'u is None, mlm_label: {mlm_label}')
            return None
        u, latent = u.to(self.device), latent.to(self.device)
        for i in range(len(u)):
            '''ordered, non-zero length'''
            full_vals = mlm_input.clone()
            full_vals[u[:i+1]] = mlm_label[u[:i+1]]
            # print(f'Inside pseudolikelihood(), full vals: \n{full_vals}, \nfull_len: {full_vals.size(0)}')
            ei_dist = self.energy( #full length processing
                idx=u[i],
                val=False, 
                rest_idx=full_vals.view(-1, 1), 
                latent=latent,
                pos_ids=torch.arange(full_vals.size(0)),
            )
            stable_dist = (-1*ei_dist) - (-1*ei_dist).max() - 1 #subtract the max value for -1*ei_dist (log-sum-exp trick)
            z_ui = torch.sum(torch.exp(stable_dist), dim=-1) #Size(1)
            expanded_z_ui = z_ui.unsqueeze(0).expand_as(stable_dist) #Size(num_classes)?
            logp_xui = torch.log(torch.exp(stable_dist) / expanded_z_ui)
            assert (stable_dist != 0).any(), f'ei_dist contains zero!!'
            assert torch.isnan(logp_xui).any() == False, f'logp_xui is NaN!! ei_dist: \n{ei_dist}, \nlatent: \n{latent}\nmlm_input: \n{mlm_input.squeeze()}\nz_ui: {z_ui}\ntorch.exp(-1*ei_dist): {torch.exp(-1*ei_dist)}\ni: {i}'
            logp_xu.append(logp_xui.unsqueeze(0)) #Eq.(0), with sum replaced by concatenation
        #end of u iter
        logp_xu = torch.cat(logp_xu, dim=0) #Size(|u|, num_classes) 
        
        return logp_xu
    
    def pseudolikelihood_old(self, latent, mlm_label, mlm_input):
        '''
        Non-batchalized! (single sample)
        
        estimte the logp using the model output logits and the MLM labels 
        params: 
            - latent: model output logits, Size(seq_len, num_classes)
            - mlm_label: input_ids for the partially masked tokens, Size(seq_len); filled with zeros
            - mlm_input: input_ids for the observed tokens, Size(seq_len); partially revealed bert_input
        return:
            - lop_xu: the conditional logprob distribution, Size(seq_len, num_classes)
        '''
        raise
        assert latent.size(0) == mlm_label.size(0) == mlm_input.size(0), \
            f'latent.shape: {latent.shape}, ' \
            f'mlm_label({mlm_label.shape}): {mlm_label}, ' \
            f'mlm_input({mlm_input.shape}): {mlm_input}'
        # print(f'Inside pseudolikelihood: latent.shape: {latent.shape}, mlm_label.shape: {mlm_label.shape}')
        # print(f'\nInside pseudolikelihood(), latent({latent.shape})\nmlm_label({mlm_label.shape}): \n{mlm_label}\nmlm_input({mlm_input.shape}): \n{mlm_input}')
        # 1. generate the unobserved and observed token indices
        u = torch.nonzero(mlm_label != self.tokenizer.pad_token_id).squeeze() #Size(|u|)
        # ignore the MASK and PAD tokens in the partially unmasked input
        o_mask = (mlm_input != self.tokenizer.mask_token_id) & (mlm_input != self.tokenizer.pad_token_id)
        o = torch.nonzero(o_mask, as_tuple=True)[0]
        # print(f'after removing padding, o({o.shape}): {o}')
        o_len = o.size(0)
        # u and o can be a single token value
        if u.dim():
            pi = torch.arange(u.size(0))
            u_prime = u
            # pi = torch.randperm(u.size(0)) #a random estimating order
            # u_prime = u[pi]
        else: # u is a single value
            pi = torch.tensor([0])
            u, u_prime = torch.tensor([u]), torch.tensor([u])
        if o.dim() == 0:
            o = torch.tensor([o])
        # except:
        #     raise IndexError(f'u: {u}. Dimension specified as 0 but tensor has no dimensions')
        logp_xu = []
        mlm_label = mlm_label.unsqueeze(-1) #Size(45,1)
        mlm_input = mlm_input.unsqueeze(-1)
        # TODO: 这一步可能有问题：batch内如果已经fully unmasked
        if len(u_prime) == 0:
            return None
        assert len(u_prime), f'u_prime is empty!! mlm_label: {mlm_label}, u: {u}, pi: {pi}'
        # print(f'\nu: {u}, \nu_prime: {u_prime}, \no: {o}')
        u_prime, o, latent = u_prime.to(self.device), o.to(self.device), latent.to(self.device)
        for i in range(len(u_prime)): #iter through |u'| EBMs
            condition_vals = mlm_label[u_prime[:i+1], :] #inclusive x_{u'_i} 121??
            condition_latent = latent[u_prime[:i+1], :]
            #note that MASK state is also included in the full_latent, the original tokens order is not maintained, but still matched
            #we found that including MASK state can give better performance? 
            '''ordered, non-zero length'''
            full_vals = mlm_input.squeeze(-1).clone()
            full_vals[u_prime[:i+1]] = mlm_label.squeeze(-1)[u_prime[:i+1]]
            inp_mask = (full_vals > self.special_tok_size)
            yo_mask = (mlm_label.view(-1) > 0).to(self.device)
            full_idx_mask = (inp_mask | yo_mask) #whether tokens are observable (the initial yo tokens are [MASK] but observable)
            
            pos_ids = (full_idx_mask).nonzero(as_tuple=True)[0] # ordered o and ui (not concatenated！) #full_vals
            full_vals = full_vals[pos_ids].unsqueeze(-1)
            full_latent = latent[pos_ids, :]
            removed_zeros = (full_vals[:u_prime[i], :].squeeze(-1) == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0] #the disgarded zeros in this partial input during nonzero()
            # print(f'\ninside pseudolikelihood(), i: {i}, removed_zeros: {removed_zeros}, full_vals: {full_vals.squeeze()}, u_prime[i]: {u_prime[i]}\n')
            removed_zeros = removed_zeros.numel()
            
            '''inordered, indexed xo+xu'''
            # full_vals = torch.cat([mlm_input[o, :], condition_vals], dim=0)
            # full_latent = torch.cat([latent[o, :], condition_latent], dim=0)
            # pos_ids = torch.cat([o.view(1, -1), u_prime[:i+1].view(1, -1)], dim=1).squeeze()
            
            # print(f'\npos_ids.shape: {pos_ids.shape}\n') #14
            # print(f'\ni: {i}, full_vals({full_vals.shape}): \n{full_vals.squeeze(-1)}\nfull_latent.shape: {full_latent.shape}')
            
            # ei_dist = self.energy(idx=pi[i], val=False, rest_idx=condition_vals, latent=condition_latent)
            # ei_dist = self.energy(idx=self.inp_len+2+pi[i], val=False, rest_idx=full_vals, latent=full_latent)
            ei_dist = self.energy(
                idx=u_prime[i]-removed_zeros, #o_len+pi[i], 
                val=False, 
                rest_idx=full_vals, 
                latent=full_latent,
                pos_ids=pos_ids,
            )
            
            # ei = self.energy(idx=pi[i], val=True, rest_idx=condition_vals, latent=condition_latent)
            # perform logits normalization to avoid nan z_ui
            max_energy = ei_dist.max() 
            ei_dist = ei_dist - max_energy
            # print(f'\nei_dist.shape: {ei_dist.shape}, ei: {ei}')
            z_ui = torch.sum(torch.exp(-1*ei_dist), dim=-1) #Size(1)
            expanded_z_ui = z_ui.unsqueeze(0).expand_as(ei_dist) #Size(num_classes)?
            logp_xui = torch.log(torch.exp(-1*ei_dist) / expanded_z_ui)
            assert torch.isnan(logp_xui).any() == False, f'logp_xui is NaN!! ei_dist: \n{ei_dist}, \nfull_latent: \n{full_latent}\nmlm_input: \n{mlm_input.squeeze()}\ni: {i}'
            # print(f'\nlogp_xui.shape: {logp_xui.shape}') #Size(num_classes)
            logp_xu.append(logp_xui.unsqueeze(0)) #Eq.(0), with sum replaced by concatenation
        #end of EBM iter
        logp_xu = torch.cat(logp_xu, dim=0) #Size(|u'|, num_classes) #logp_xu_prime
        # # TODO: 注意，这里是按u_prime顺序cat的，还需要恢复顺序
        # pi_inv = torch.empty_like(pi)
        # pi_inv[pi] = torch.arange(pi.size(0))
        # logp_xu = logp_xu_prime[pi_inv]
        
        return logp_xu
    
class GPTSequentialEBMs():
    '''
    Use a GPT2-style model from scratch, allows flexible input and output lengths
    ''' 
    def __init__(self, tokenizer, task_config, model_config=None, device='cpu'):
        self.task_name = task_config.name
        self.param_type = 'gpt'
        # print(f'Inside init SEBM, cwd: {os.getcwd()}')
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.vocab_size = task_config.num_classes
        self.device = device ###########for debugging
        self._build_model(model_config)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def _build_model(self, model_info):
        if isinstance(model_info, str): #TODO
            self.model = AutoModelForCausalLM(model_info)
        elif isinstance(model_info, dict):
            raise
        elif model_info == None:
            gpt_config = AutoConfig.from_pretrained('./ire_reasoning/models/model_config_tiny')
            gpt_config.n_positions = self.task_config.max_len #output_len?
            print(f'model max_seq_len: {gpt_config.n_positions}')
            self.model = AutoModelForCausalLM.from_config(gpt_config) #not ebm!
            self.d_model = gpt_config.n_embd #determines the initial lr
        else:
            self.model = model_info
            self.d_model = model_info.config.n_embd #determines the initial lr
        self.model.to(self.device)
    
    def energy(self, idx:int, val: bool, rest_idx: torch.Tensor, latent: torch.Tensor, \
        batchalize=False) -> torch.Tensor:
        '''
        E(x_ui (=val) ; rest_idx)
        
        Given a specified dimension, an index table and a latent value source, 
        calculate the energy value for the specified index value on that dimension, 
        or all the energy values along that dimension.
        
        params:
            idx: the specified index "i" to predict
            val: whether the predict value "x_{u'_i}" is given, determines what to return
            rest_idx: the conditional values (include pos i!), represented by a 2D tensor of Size(|x_{u'_{<=i}}|, 1)
            latent: the value source for energy calculation, represented by a 2D tensor of Size(|x_{u'_{<=i}}|, num_classes) 
            
        return:
                energy: Size(1), if specified val;
                energy vector: Size(vocab_size), otherwise
        '''
        assert latent.dim() == rest_idx.dim() and latent.size(-2) == rest_idx.size(-2), \
            f"latent.shape = {latent.shape}, rest_idx.shape = {rest_idx.shape}"
        if val:
            energy = torch.gather(input=latent, dim=-1, index=rest_idx)
        else: #energy dist
            if batchalize:
                expanded_idx = torch.zeros(rest_idx.size(0), rest_idx.size(1), self.vocab_size)
                for pos in range(expanded_idx.size(1)):
                    if pos == idx: #enumerate all vals on ui
                        expanded_idx[:, pos, :] = torch.arange(self.vocab_size) 
                    else: #fill the original val on other positions for each class
                        expanded_idx[:, pos, :] = rest_idx[:, pos, :].expand(-1, self.vocab_size) 
            else:   
                expanded_idx = torch.zeros(rest_idx.size(0), self.vocab_size)
                for pos in range(expanded_idx.size(0)): #|u'i|
                    if pos == idx: #enumerate all vals on ui
                        expanded_idx[pos, :] = torch.arange(self.vocab_size) 
                    else: #fill the original val on other positions for each class
                        expanded_idx[pos, :] = rest_idx[pos, :].expand(self.vocab_size) 
                # print(f'\nrest_idx: \n{rest_idx}, \nexpanded_idx: \n{expanded_idx}')
            expanded_idx = expanded_idx.to(torch.long).to(self.device)
            energy = torch.gather(input=latent, dim=-1, index=expanded_idx) #Size(out_len, num_classes)
                
        return -1 * torch.sum(energy, dim=-2) #sum along all ui positions #-1 *

    def gibbs_dist(self, energy_dist: torch.Tensor, energy_clip=True):
        '''
        Given the energy distribution at position i across all classes,
        calculate the Boltzmann (Gibbs) distribution. 
        
        params:
            energy_dist: Size(batch_size, num_classes)
            energy_clip: whether subtract the maximum energy before calculating the p_i distribution (to waive NaN), default to True
        return:
            p_i: Size(batch_size, num_classes)
        '''
        if energy_clip: 
            e_max = energy_dist.max()
            energy_dist = energy_dist - e_max
        
        z_i = torch.sum(torch.exp(-1*energy_dist), dim=-1) #Size(batch_size)
        expanded_zi = z_i.unsqueeze(1).expand_as(energy_dist) #Size(batch_size, num_classes)
        # Sample from the 1D conditional p(y_{o_i} | y_{o_-i})
        p_i = torch.exp(-1*energy_dist) / expanded_zi #Size(batch_size, num_classes)
        
        return p_i
    
    def sampling(self, order_label:int, partial_pred: torch.Tensor, sample_batch:dict, \
        sampler='gibbs', sampling_times=10, visual_ebms=None):
        '''
        Sampling on a partially masked sample batch.
        
        params: 
            order_label: k (1-indexed)
            partial_pred: Size(batch_size, seq_len)
            sample_batch: dict
            samping_times: T
            
        return:
            updated_partial_pred: Size(batch_size, seq_len)
            sth: dict, contains loss_list, energy_list, visual_ebms.log, or testing infos 
        '''
        # 1. Initialize yo and inputs to the model.forward and energy calcualtion
        batch_size = sample_batch['input'].size(0)
        losses = []
        energies = []
        pred = []
        if sampler == 'gibbs':
            for b in range(batch_size):
                previous_pred = partial_pred[b]
                yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                    (sample_batch['schedule_label'][b] <= order_label)).nonzero(as_tuple=True)[0].to(self.device)
                # print(f'yo_idx.shape({yo_idx.shape}): \n{yo_idx}') #Size(|o|)
                # print(f'previous_pred.device: {previous_pred.device}, yo_idx.device: {yo_idx.device}, self.device: {self.device}')
                yo = previous_pred[yo_idx]
                # print(f'yo({yo.shape}): \n{yo}')
                model_input = sample_batch['input'][b].clone()
                label = sample_batch['input'][b].clone()
                label[yo_idx] = sample_batch['label'][b][yo_idx]
                # # Initialize Method1: fully random
                # model_input[model_input == 3] = 0 #replace the MASK with 0
                # model_input += previous_pred
                # Initialize Method2: keep the history prediction from previous k-1 iterations
                if order_label != 1:
                    history_yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                        (sample_batch['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
                    model_input[history_yo_idx] = previous_pred[history_yo_idx]
                # print(f"\norder_label={order_label}, bert_input: {sample_batch['bert_input'][b]}\nprevious_pred: {previous_pred}"\
                #     f"\nmodel_input: {model_input}")
                for t in range(sampling_times):
                    # print(f'\n____\nStart t={t}-th sampling...\n')
                    # print(f'forward input.shape: {model_input.unsqueeze(0)}')
                    forward_input = {
                        'input': model_input.unsqueeze(0),
                        'label': label.unsqueeze(0),
                    }
                    gamma = self.forward(forward_input, \
                        None, is_ebm=True).view(-1, self.vocab_size) # sample_batch['segment_label'][b].unsqueeze(0)
                    # print(f'after reshape, gamma({gamma.shape})') #30,6
                    # print(f'\nenergy inputs: rest_idx.shape={yo.unsqueeze(-1)}, latent.shape={gamma[yo_idx, :].shape}')
                    yo_energy = self.energy(idx=0, val=True, rest_idx=yo.unsqueeze(-1), \
                        latent=gamma[yo_idx, :], batchalize=False)
                    # print(f'yo_energy: {yo_energy}')
                    
                    if b == 0 and t == 0:
                        energies.append(round(yo_energy.item(), 2)) #initial energy
                    
                    # 2. gibbs sampling on each masked position
                    yo_prime = yo.clone()
                    for i in range(yo_idx.size(0)): #iter |o|
                        # sample on position i
                        # print(f'inputs to the energy: yo_prime: {yo_prime}, gamma[yo_idx,:]: {gamma[yo_idx, :]}')
                        ei_dist = self.energy(idx=i, val=False, rest_idx=yo_prime.unsqueeze(-1), \
                            latent=gamma[yo_idx, :], batchalize=False)
                        p_oi = self.gibbs_dist(ei_dist.unsqueeze(0)) #Size(1,6)
                        
                        #_________forcing ignoring/considering the special tokens_______
                        # print(f'p_oi({p_oi.shape}): \n{p_oi},\n ei_dist({ei_dist.shape}): \n{ei_dist}')
                        y_oi_prime = torch.multinomial(p_oi[:,self.special_tok_size:], 1) + self.special_tok_size
                        # y_oi_prime = torch.multinomial(p_oi, 1)
                        #___________________________________________________
                        
                        # update the sampled  yo'_i to yo'
                        yo_prime[i] = y_oi_prime.squeeze()
                        # print(f'i={i}: \n- y_oi\': {y_oi_prime.item()}, \n- ei_dist: {ei_dist}, \n- logits: {gamma[yo_idx[i], :]}')
                    # 3. update yo with yo' if the energy decreases
                    yo_prime_energy = self.energy(idx=0, val=True, rest_idx=yo_prime.unsqueeze(-1), \
                        latent=gamma[yo_idx, :], batchalize=False)
                    # print(f'yo\' energy: {yo_prime_energy}')
                    if yo_prime_energy.item() < yo_energy.item():
                        yo = yo_prime
                        # update model input as well
                        # print(f'yo\' energy is smaller, before update, previous_pred: {previous_pred}')
                        previous_pred[yo_idx] = yo #?
                        # print(f'after update, previous_pred: {previous_pred}, \nbert_input: {sample_batch["bert_input"][b]}')
                        model_input = sample_batch['input'][b].clone()
                        model_input[previous_pred != 0] = previous_pred[previous_pred != 0]
                        # model_input[model_input == (self.special_tok_size-1)] = 0
                        # model_input += previous_pred
                        # print(f'model_input: {model_input}')
                    # 4. Record partial prediction, losses(using logits) and energies (last sample in the batch)
                    if b == 0:
                        loss = self.criterion(gamma[yo_idx, :], sample_batch['label'][b][yo_idx])
                        losses.append(round(loss.item(),2))
                        energies.append(round(yo_energy.item(),2))
                #end of t iter
                pred.append(previous_pred.view(1,-1))
                # break ####################test
            #end of inner-batch iter
        #end of 'gibbs'
        elif sampler == 'argmin_energy': 
                '''
                similar to 'sft', but in AR style
                (due to sequence of k, though history tokens might change) #greedy optimize
                '''
                for b in range(batch_size):
                    previous_pred = partial_pred[b]
                    yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                        (sample_batch['schedule_label'][b] <= order_label)).nonzero(as_tuple=True)[0] #<=
                    # if b == 0:
                    #     print(f'k: {order_label}, yo_idx({yo_idx.shape}): \n{yo_idx}') #Size(|o|)
                    # yo = previous_pred[yo_idx]
                    # print(f'yo({yo.shape}): \n{yo}')
                    model_input = sample_batch['input'][b].clone()
                    # Initialize Method1: fully random initialize (wrong) 
                    # model_input[model_input == 3] = 0 #replace the MASK with 0
                    # model_input += previous_pred
                    
                    # Initialize Method2: keep the history prediction from previous k-1 iterations
                    if order_label != 1:
                        history_yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                            (sample_batch['schedule_label'][b] < order_label)).nonzero(as_tuple=True)[0]
                        model_input[history_yo_idx] = previous_pred[history_yo_idx]
                    
                    # forward pass with softmax in a single run (gamma.size = (30,6))
                    gamma = self.model.forward(model_input.unsqueeze(0), \
                            None, is_ebm=False)[0].view(-1, self.vocab_size) # sample_batch['segment_label'][b].unsqueeze(0)
                    gamma[:, 0] = IGNORE_INDEX
                    # print(f'gamma.shape: {gamma.shape}') #torch.Size([50, 31])
                    yo = gamma[yo_idx, :].argmax(dim=-1) #same as argmin_energy #argmin for '_w_mask.pth'
                    # print(f'b={b}, yo({yo.shape}): {yo}')
                    previous_pred[yo_idx] = yo
                    pred.append(previous_pred.unsqueeze(0)) #view(1,-1)
                    # record loss and energy (first sample per batch)
                    if b == 0:
                        # print(f'k={order_label}, gamma: \n{gamma}')
                        loss = self.criterion(gamma[yo_idx, :], sample_batch['label'][b][yo_idx])
                        yo_energy = self.energy(idx=0, val=True, rest_idx=yo.unsqueeze(-1), \
                            latent=gamma[yo_idx, :], batchalize=False)
                        losses.append(round(loss.item(),2))
                        energies.append(round(yo_energy.item(),2))
                # end of inner-batch iter
        pred = torch.cat(pred, dim=0)
        # print(f'sampled pred[0] ({pred.shape}): {pred[0]}')
        
        return pred, {'losses': losses, 'energies': energies}

    def pseudolikelihood(self, latent, mlm_label, mlm_input):
        '''
        Non-batchalized! (single sample)
        
        estimte the logp using the model output logits and the MLM labels 
        params: 
            - latent: model output logits, Size(seq_len, num_classes)
            - mlm_label: input_ids for the partially masked tokens, Size(seq_len); filled with zeros
            - mlm_input: input_ids for the observed tokens, Size(seq_len); partially revealed bert_input
        return:
            - lop_xu: the conditional logprob distribution, Size(seq_len, num_classes)
        '''
        assert latent.size(0) == mlm_label.size(0) == mlm_input.size(0), \
            f'latent.shape: {latent.shape}, ' \
            f'mlm_label({mlm_label.shape}): {mlm_label}, ' \
            f'mlm_input({mlm_input.shape}): {mlm_input}'
        # print(f'Inside pseudolikelihood: latent.shape: {latent.shape}, mlm_label.shape: {mlm_label.shape}')
        # print(f'\nInside pseudolikelihood(), latent({latent.shape})\nmlm_label({mlm_label.shape}): \n{mlm_label}\nmlm_input({mlm_input.shape}): \n{mlm_input}')
        # 1. generate the unobserved and observed token indices
        u = torch.nonzero(mlm_label != self.tokenizer.pad_token_id).squeeze() #Size(|u|)
        # ignore the MASK and PAD tokens in the partially unmasked input
        o_mask = (mlm_input != self.tokenizer.mask_token_id) & (mlm_input != self.tokenizer.pad_token_id)
        o = torch.nonzero(o_mask, as_tuple=True)[0]
        # print(f'after removing padding, o({o.shape}): {o}')
        o_len = o.size(0)
        # u and o can be a single token value
        if u.dim():
            pi = torch.arange(u.size(0))
            u_prime = u
            # pi = torch.randperm(u.size(0)) #a random estimating order
            # u_prime = u[pi]
        else: # u is a single value
            pi = torch.tensor([0])
            u, u_prime = torch.tensor([u]), torch.tensor([u])
        if o.dim() == 0:
            o = torch.tensor([o])
        # except:
        #     raise IndexError(f'u: {u}. Dimension specified as 0 but tensor has no dimensions')
        logp_xu = []
        mlm_label = mlm_label.unsqueeze(-1) #Size(45,1)
        mlm_input = mlm_input.unsqueeze(-1)
        # TODO: 这一步可能有问题：batch内如果已经fully unmasked
        if len(u_prime) == 0:
            return None
        assert len(u_prime), f'u_prime is empty!! mlm_label: {mlm_label}, u: {u}, pi: {pi}'
        # print(f'\nu: {u}, \nu_prime: {u_prime}, \no: {o}')
        for i in range(len(u_prime)): #iter through |u'| EBMs
            condition_vals = mlm_label[u_prime[:i+1], :] #inclusive x_{u'_i} 121??
            condition_latent = latent[u_prime[:i+1], :]
            #note that MASK state is also included in the full_latent, the original tokens order is not maintained, but still matched
            #we found that including MASK state can give better performance? 
            '''ordered, full length'''
            # full_vals, full_latent = mlm_input.clone(), latent
            # full_vals[u_prime[:i+1], :] = condition_vals
            '''inordered, indexed xo+xu'''
            full_vals = torch.cat([mlm_input[o, :], condition_vals], dim=0)
            full_latent = torch.cat([latent[o, :], condition_latent], dim=0)
            # print(f'\ni: {i}, full_vals({full_vals.shape}): \n{full_vals.squeeze(-1)}\nfull_latent.shape: {full_latent.shape}')
            
            # ei_dist = self.energy(idx=pi[i], val=False, rest_idx=condition_vals, latent=condition_latent)
            # ei_dist = self.energy(idx=self.inp_len+2+pi[i], val=False, rest_idx=full_vals, latent=full_latent)
            ei_dist = self.energy(idx=o_len+pi[i], val=False, rest_idx=full_vals, latent=full_latent)
            
            # ei = self.energy(idx=pi[i], val=True, rest_idx=condition_vals, latent=condition_latent)
            # perform logits normalization to avoid nan z_ui
            max_energy = ei_dist.max() 
            ei_dist = ei_dist - max_energy
            # print(f'\nei_dist.shape: {ei_dist.shape}, ei: {ei}')
            z_ui = torch.sum(torch.exp(-1*ei_dist), dim=-1) #Size(1)
            expanded_z_ui = z_ui.unsqueeze(0).expand_as(ei_dist) #Size(num_classes)?
            logp_xui = torch.log(torch.exp(-1*ei_dist) / expanded_z_ui)
            assert torch.isnan(logp_xui).any() == False, f'logp_xui is NaN!! ei_dist: \n{ei_dist}, \nexpanded_z_ui: \n{expanded_z_ui}'
            # print(f'\nlogp_xui.shape: {logp_xui.shape}') #Size(num_classes)
            logp_xu.append(logp_xui.unsqueeze(0)) #Eq.(0), with sum replaced by concatenation
        #end of EBM iter
        logp_xu = torch.cat(logp_xu, dim=0) #Size(|u'|, num_classes) #logp_xu_prime
        # # TODO: 注意，这里是按u_prime顺序cat的，还需要恢复顺序
        # pi_inv = torch.empty_like(pi)
        # pi_inv[pi] = torch.arange(pi.size(0))
        # logp_xu = logp_xu_prime[pi_inv]
        
        return logp_xu
    
    def forward(self, input_dict, segment_label=None, is_ebm=False, no_grad=False): #returns logits and loss??
        # from undecorated import undecorated
        from types import MethodType
        # TODO: check trainer class?
        '''
        input_dict: batched dict {input_ids, attention_mask, src_mask, (label?不存在?)}
        '''
        batch_size = input_dict['input'].size(0)
        logits = []
        for b in range(batch_size):
            # remove paddings
            # print(f'before remove paddings, input.shape: {input_dict["input"][b].shape}')
            input_ids = input_dict['input'][b] != self.tokenizer.pad_token_id
            label_ids = input_dict['label'][b] != self.tokenizer.pad_token_id
            input = input_dict['input'][b]#[input_ids]
            label = input_dict['label'][b]#[label_ids]
            attention = torch.tensor([1]*(input.size(0)), device=self.device)
            # model_input = {
            #     input_ids: input,
            #     attention: attention,
            # }
            # print(f'\nafter removing paddings, b={b}, input.shape: {input.shape}, label.shape: {label.shape}')
            # generate
            # output_dict = self.model(*model_input) #TypeError: iteration over a 0-d tensor
            output_dict = self.model.forward( #loss, logits (4, 512, 31), past_key_values
                input_ids=input,
                attention_mask=attention, #label没用上
                labels=label
            ) # output_dict.logits.shape: Size(50, 31) #full_len
            logits.append(output_dict.logits.unsqueeze(0))
        logits = torch.cat(logits, dim=0)
        if is_ebm:
            return logits #gamma, mlm_output
        else: #return output ids (argmax, if not do_sample)
            return self.softmax(logits), None

'''
Batchalized version of BERTSequentialEBMs. Hope for better performance.

Features: 
- new sebm(base_model): original BERT encoder + classifier decoder
- batchalized pseudolikelihood, losses calcualtion
'''
class FastSequentialEBMs():
    def __init__(self, tokenizer, task_config, d_model, n_layers, heads=8, device='cpu'):
        self.param_type = 'fast' #generally the same as BERT, with fast pseudolikelihood and contrast loss calculation
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.task_name = task_config.name
        self.max_len = task_config.max_len
        self.vocab_size = task_config.num_classes
        self.special_token_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.mask_token_id,
            self.tokenizer.unk_token_id,
        }
        self.special_tok_size = len(self.special_token_ids)
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        self._build_model()
        print(f'Initializing FastSequentialEBMs is completed!')
    
    def _build_model(self):
        self.model = EnergyBERT(
            vocab_size = self.vocab_size,
            max_len = self.max_len,
            hidden_size = self.d_model,
            n_layers = self.n_layers,
            heads = self.heads,
        )
        self.model.to(self.device)
    
    def get_2D_indices(self, array, val, type='remove_pad'): #n is the number of wanted elements in a row
        '''
        Helper function to get the indices of particular elements in a 2D manner.
        
        types: 
            remove_pad: return the incides of the non-padding values (val is the padding value)
            prev_unmask: return the output tokens which have scheduling label < current scheduling label(specified by val)
            curr_unmask: return the output tokens which have shceduling label == current shceduling label(specified by val)
            full_unmask: return the output tokens which have scheduling label <= current scheduling label(specified by val)
        '''
        row_size, col_size = array.shape
        if type == 'remove_pad':
            coords = torch.nonzero(array != val, as_tuple=False)
            n = torch.count_nonzero(array[0] != val) #number of non-val elements per sample
        elif type == 'prev_unmask':
            coords = torch.nonzero(array < val, as_tuple=False)
            n = torch.count_nonzero(array[0] < val)
        elif type == 'curr_unmask': 
            coords = torch.nonzero(array == val, as_tuple=False)
            n = torch.count_nonzero(array[0] == val)
        elif type == 'full_unmask':
            coords = torch.nonzero(array <= val, as_tuple=False)
            n = torch.count_nonzero(array[0] <= val)
        assert coords.dim() == 2
        row_ids, col_ids = coords[:, 0], coords[:, 1]
        ids = torch.empty(row_size, n, device=self.device)
        for r in range(row_size):
            ids[r, :] = col_ids[row_ids == r]
        return ids.long()
    
    # wrapper function, return in Size(batch_size)
    def energy(self, batched_input_ids):
        batched_input_ids[batched_input_ids == -1] = 1
        try:
            return self.model.forward(batched_input_ids).squeeze() #Size(batch_size, 1) -> Size(batch_size)
        except:
            raise IndexError(f'forward input({batched_input_ids.shape}): {batched_input_ids}')
        
    def pseudolikelihood_revised(self, xo, xu):
        '''
        Temporary revised version of pseudolikelihood_batched.
        params:
            - xo: (B, full_len), the sequence of input and previously unmasked output tokens
            - xu: (B, full_len), zero-padded output tokens to be unmasked at the current timestamp (为了节省memory并没有重新repredict historical unmasking tokens)
        returns:
            - logp_xu: (B, V, U): distribution of the "pseudo-logits"
            - energy_dist: (B, V, U): the energy landscape spanned by U and V
        '''
        # torch.set_printoptions(threshold=float('inf'))   # or a huge integer
        # 1. Get all the sizes needed
        # if self.device == 0: ####test
        #     print(f'at the start of pseudolikelihood calculation, \n{torch.cuda.memory_summary()}')
        u = self.get_2D_indices(
            array=xu,
            val=self.tokenizer.pad_token_id,
            type='remove_pad',
        )
        xu_ = xu.clone()
        xu_[xu_ == self.tokenizer.unk_token_id] = self.tokenizer.pad_token_id
        B, full_len = xo.shape
        device = self.device
        U, V = u.size(1), self.vocab_size
        # 2. Initialize the all-in x_base as input for batchalized energy calculation
        x_base = xo.clone().unsqueeze(1).unsqueeze(1).expand(B, U, V, full_len) #later to be varied on U and V dimensions
        # if self.device == 0: ####test
        #     print(f'x_base is initialized, \n{torch.cuda.memory_summary()}')
        # if U == 2:
        #     print(f'\nInside pseudolikelihood(), xo({xo.shape}): \n{xo},\nxu({xu.shape}): \n{xu}')
        #     print(f'u({u.shape}): {u}')
        #     print(f'x_base initialized has shape: {x_base.shape}')
        # 3. Unmask earlier positions j < i for every i (vary on U before i)
        if U > 1:
            mask_early = torch.arange(U, device=device).unsqueeze(0) < torch.arange(U, device=device).unsqueeze(1) #(U,U), downward triangular matrix of True's among False's
            mask_early = mask_early.unsqueeze(0).unsqueeze(2).expand(B,U,V,U) #(B_,U,V_,U)
            # if U == 2:
            #     print(f'mask_early({mask_early.shape}): {mask_early}')
            idx_early = u.unsqueeze(1).unsqueeze(2).expand(B,U,V,U) #(B,U_,V_,U)
            u_early = xu_.gather(dim=-1, index=u).unsqueeze(1).unsqueeze(2).expand(B,U,V,U) #(B,U_,V_,U)
            o_early = xo.gather(dim=-1, index=u).unsqueeze(1).unsqueeze(2).expand(B,U,V,U) #(B,U_,V_,U)
            val_early = torch.where(mask_early, u_early, o_early) #(B,U,V_,U)
            # if U == 2:
            #     print(f'val_early({val_early.shape}): {val_early}')
            x_base = x_base.scatter(dim=-1, index=idx_early, src=val_early.long())
            # if U == 2:
            #     print(f'x_base with unmasked positions earlier than i ({x_base.shape}): \n{x_base}')
        # 3. Write class value v at position u[b, i] for every v
        idx_u = u.unsqueeze(2).expand(B,U,V).unsqueeze(-1) #B,U,V,1
        val_v = torch.arange(V, device=device).view(1,1,V,1).expand(B,U,V,1) #B,U,V,1
        x_base = x_base.scatter(dim=-1, index=idx_u, src=val_v.long())
        # if U == 2:
        #     print(f'x_base with varied v ({x_base.shape}): \n{x_base}')
        # 4. Flatten the x_base and compute the energies in a single forward (might result in OOM!)
        flat = x_base.reshape(-1, full_len) #(B*U*V,full_len)
        # if U == 2:
        #     print(f'after falttened for energy() to take in, flat({flat.shape}): \n{flat}')
        # if self.device == 0: ####test
        #     print(f'before energy calculation, \n{torch.cuda.memory_summary()}')
        # if U == 2:
        flat_energy = self.energy(flat) #(B*U*V)
        energy_dist = flat_energy.view(B,U,V) #B,U,V
        # if U == 2:
        #     print(f'the final energy_dist({energy_dist.shape}): \n{energy_dist}')
        # 5. Gibbs distribution
        log_gibbs_dist = self.log_gibbs_dist(energy_dist)
        # if U == 2:
        #     print(f'the logp_xu before permuting ({pseudolikelihood.shape}): \n{pseudolikelihood}')
        
        return log_gibbs_dist.permute(0,2,1).contiguous(), energy_dist.permute(0,2,1).contiguous()
            
    def pseudolikelihood_batched(self, xo, xu):
        '''
        More batchalized version of pseudolikelihood for reducing the for-loop iteration on V.
        
        returns: logp_xu in Size(batch_size, vocab_size, |u|)
        '''
        B, full_len = xo.shape
        device = self.device
        u = self.get_2D_indices(
            array=xu,
            val=self.tokenizer.pad_token_id,
            type='remove_pad',
        )
        xu_  =xu.clone()
        xu_[xu == self.tokenizer.unk_token_id] = self.tokenizer.pad_token_id
        U, V = u.size(1), self.vocab_size
        torch.set_printoptions(threshold=float('inf'))   # or a huge integer
        if U == 2:
            print(f'\nInside pseudolikelihood(), xo({xo.shape}): \n{xo},\nxu({xu.shape}): \n{xu}')
            print(f'u({u.shape}): {u}')
        # 1. Build the input sequence for calculating the energy_dist efficiently
        x_base = xo.clone().unsqueeze(1).unsqueeze(1).expand(B,U,V,full_len)
        # 2. Unmask earlier positions j < i for every i
        if U > 1:
            mask_early = torch.arange(U, device=device).unsqueeze(0) < torch.arange(U, device=device).unsqueeze(1) #Size(U,U)的上行和右列全部False、左下三角区域全部True的 u_i mask matrix
            # print(f'mask_early({mask_early.shape}): {mask_early}')
            mask_early = mask_early.unsqueeze(0).unsqueeze(2) #1,u,1,u
            idx_early = u.unsqueeze(1).expand(B,U,U) #B,U,U
            # effect of val_early = xu_.gather(dim=1, index=idx_early) #B,U,U
            flat_idx = idx_early.reshape(-1, U)                   # (B*U, U)
            flat_xu  = xu_.unsqueeze(1).expand(-1, U, -1).reshape(-1, full_len)  # (B*U, full_len)
            val_early = flat_xu.gather(dim=1, index=flat_idx)  # (B*U, U)
            val_early = val_early.view(B, U, U)                # (B, U, U)
            if U ==2:
                print(f'val_early({val_early.shape}): \n{val_early}')
                print(f'val_early.unsqueeze(-1).expand(B, U, U, 3)({val_early.unsqueeze(-1).expand(B, U, U, 3).shape}): \n{val_early.unsqueeze(-1).expand(B, U, U, 3)}')
            # effect of x_base.scatter_(dim=1,index=idx_early.unsqueeze(2).expand(B,U,V,U),src=val_early.unsqueeze(2).expand(B,U,V,U))
            b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, U, U, V)
            i_idx = torch.arange(U, device=device).view(1, U, 1, 1).expand(B, U, U, V)
            v_idx = torch.arange(V, device=device).view(1, 1, 1, V).expand(B, U, U, V)
            target_pos = idx_early.unsqueeze(-1).expand(B, U, U, V)  # (B, U, U, V)
            x_base[b_idx, i_idx, v_idx, target_pos] = val_early.unsqueeze(-1).expand(B, U, U, V)
            if U == 2:
                print(f'after unmasked early positions before u_i, x_base({x_base.shape}): \n{x_base}')
        # 3. Write class value v at position u[b, i] for every v
        # idx_u = u.unsqueeze(2).expand(B,U,V).unsqueeze(-1) #B,U,V,1
        # val_v = torch.arange(V, device=device).view(1,1,V,1).expand(B,U,V,1) #B,U,V,1
        # effect of x_base.scatter_(dim=-1, index=idx_u, src=val_v.long())
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, U, V)
        i_idx = torch.arange(U, device=device).view(1, U, 1).expand(B, U, V)
        v_idx = torch.arange(V, device=device).view(1, 1, V).expand(B, U, V)
        u_flat = u.unsqueeze(2).expand(B, U, V)
        x_base[b_idx, i_idx, v_idx, u_flat] = v_idx.long()
        if U == 2:
            print(f'after unmasked u_i with all v values, x_base({x_base.shape}): \n{x_base}')
        # 4. Flatten and compute the energies in one forward pass
        flat = x_base.reshape(-1, full_len) #B*U*V,full_len
        if U == 2:
            print(f'after falttened for energy() to take in, flat({flat.shape}): \n{flat}')
        flat_energy = self.energy(flat) #B*U*V
        energy_dist = flat_energy.view(B,U,V) #B,U,V
        if U == 2:
            print(f'the final energy_dist({energy_dist.shape}): \n{energy_dist}')
        # 5. Gibbs distribution
        pseudolikelihood = self.log_gibbs_dist(energy_dist)
        if U == 2:
            print(f'the logp_xu before permuting ({pseudolikelihood.shape}): \n{pseudolikelihood}')
        
        return pseudolikelihood.permute(0,2,1).contiguous()
    
    def pseudolikelihood(self, xo, xu):
        '''
        Sequential EBMs' alternative method of generating the "logits"
        
        input: 
            xo: Size(batch_size, full_len), x_o with previously unmasked xu's (if any).
            xu: Size(batch_size, full_len), the zero-padded positive label for the current time stamp (k).
        returns:
            logp: Size(batch_size, vocab_size, |u|), permuted in channel-first style, ready for loss calculation
        '''
        # 1. Calculate the energy distribution
        u = self.get_2D_indices(
                    array=xu,
                    val=self.tokenizer.pad_token_id,
                    type='remove_pad'
                )
        xu_ = xu.clone()
        xu_[xu == self.tokenizer.unk_token_id] = self.tokenizer.pad_token_id
        energy_dist = torch.empty(u.size(0), u.size(1), self.vocab_size, device=self.device) #"logits": Size(batch_size, |u|, vocab_size)
        for i in range(u.size(1)): # TODO: 能否想办法batchalize?
            for v in range(self.vocab_size): #这里用完整的vocab,不用task-specific：range(self.special_tok_size, self.special_tok_size+len(self.tokenizer.vocab))
                # 1. prepare the partially unmasked sample batches for calculating the energy
                x_oui = xo.clone()
                if i: # effect of: x_oui[u[:, :i]] = xu_[u[:, :i]]
                    unmask_before_i = xu_.gather(dim=1, index=u[:, :i])
                    x_oui.scatter_(dim=1, index=u[:, :i], src=unmask_before_i)  
                # print(f'i: {i}, u[:, i:i+1]: {u[:, i:i+1]}\nv: {v}')
                x_oui.scatter_(dim=1, index=u[:, i:i+1], # effect of: x_oui[u[:, i:i+1]] = v
                    src=torch.full_like(u[:, i:i+1], v, dtype=x_oui.dtype))
                # print(f'xu_: {xu_}, \nx_oui: {x_oui}')
                
                # 2. calculate the batchalized energy.
                energy_iv = self.energy(x_oui) #Size(batch_size)
                energy_dist[:, i, v] = energy_iv
                # print(f'after calculate on v={v}, energy_dist: \n{energy_dist}\n')
                
        # 2. Calculate the log of the Gibbs distribution
        # print(f'\nenergy_dist({energy_dist.shape}): {energy_dist}')
        pseudolikelihood = self.log_gibbs_dist(energy_dist)
        assert torch.isnan(pseudolikelihood).any() == False, f'\nPseudolikelihood contains NaN!!'
        # print(f'the final pseudolikelihood.shape: {pseudolikelihood.shape}') #Size(batch_size, |u|, vocab_size)
        # move the class dimension ahead to fit the channel-first requirement for calculating the losses
        pseudolikelihood = pseudolikelihood.permute(0, 2, 1).contiguous()
        # print(f'inside pseudo function, after permute, pseudolikelihood.shape: {pseudolikelihood.shape}')

        return pseudolikelihood
    
    def randomize(self, val, actual_vocab_size):
        '''
        helper function to create a fully randomized tensor of the same size as the input.
        '''
        start_id = self.special_tok_size
        end_id = start_id + actual_vocab_size
        full_noise = torch.randint(start_id, end_id, val.size(), dtype=val.dtype, device=val.device)
        same_mask = (full_noise == val).to(val.device)
        while same_mask.any(): #Guarantee each corresponding element is different from the original one
            regenerated_noise = torch.randint(start_id, end_id, val.size(), dtype=val.dtype, device=val.device)
            full_noise[same_mask] = regenerated_noise[same_mask]
            same_mask = (full_noise == val).to(val.device)
        return full_noise
        
    def make_negative(self, u_val, energy_dist):
        '''
        Given the tokens to be unmasked, create a negative version of it.
        Negative samples have a random number of [1, |u|] tokens picked at random output positions,
        anf have their value shifted by +-1. 
        
        input: 
            u_val: (B,U), the tokens to be unmaked from the positive sample batch
            energy_dist: (B,V,U), the energy landscape spanned by V and U
        return:
            neg_u_val: (B,U), the noise-added tokens
        '''
        (B,U), device = u_val.size(), self.device
        # 1. get the u_vals corresponding to the minial and the second minimal energy
        # vals, ids = torch.topk(energy_dist, k=2, dim=1, largest=False) #both (B,2,U)
        # full_noise = ids[:, 1, :] #(B,U) #这个negative sample生成办法不顶用
        full_noise = self.randomize(u_val, len(self.tokenizer.vocab))
        # 2. get a noise position mask
        pos_one = torch.randint(0, U, (B,1), device=device)
        mask1 = torch.zeros(B,U, dtype=torch.bool, device=device)
        mask1.scatter_(dim=1, index=pos_one, src=torch.ones(B,1,dtype=torch.bool,device=device)) #at least one noised position
        num_others = torch.randint(0, B*U+1, (1,), device=self.device)
        flat_idx = torch.randperm(B*U, device=self.device)[:num_others]
        mask2 = torch.zeros(B*U, dtype=torch.bool, device=self.device)
        mask2[flat_idx] = True
        mask2 = mask2.view(B, U) #other randomly noised positions
        noise_mask = mask1 | mask2
        # 3. get the negative u_val where all non-golden values has the second minimal energy
        neg_u_val = torch.where(noise_mask, full_noise, u_val)
        # print(f'neg_u_val.shape: {neg_u_val.shape}')
        # print(f'\nnoise_mask: \n{noise_mask}, \nu_val: \n{u_val}, \nneg_u_val: \n{neg_u_val}')
        return neg_u_val.to(device)
    
    def contrast_loss_old(self, energy_dist, xu, loss_type='l2', threshold=2):
        '''
        Revised version of fast_contrast_loss().
        inputs:
            - energy_dist: (B,V,U), the energy landscape spanned by V and U, an intermediate result from pseudolikelihood()
            - xu: (B, full_len), the zero-padded label for specific k
        returns:
            - contrast_loss: Size(1), scalar L2 contrast
        '''
        B, V, U = energy_dist.size()
        u = self.get_2D_indices(
            array=xu,
            val=self.tokenizer.pad_token_id,
            type='remove_pad'
        ) #(B,U)
        assert u.size(0)==B and u.size(1)==U, f'u.shape: {u.shape}, B: {B}, V: {V}, U: {U}'
        # 1. get the (B,U) postive token values at the current step k
        pos_u_val = xu.clone().gather(dim=1, index=u) #(B,U)
        # print(f'pos_u_val({pos_u_val.shape}): {pos_u_val}')
        # 2. get the corresponding corrupted negative token values of (B,U)
        neg_u_val = self.make_negative(pos_u_val, energy_dist)
        # 3. index the calculated energy landcape with positive and negative token value
        pos_energy = energy_dist.gather(dim=1, index=pos_u_val.unsqueeze(1)).squeeze(1).sum(dim=1) #(B,)
        neg_energy = energy_dist.gather(dim=1, index=neg_u_val.unsqueeze(1)).squeeze(1).sum(dim=1)
        # 4. calculate the L2 contrast loss
        if loss_type == 'l2':
            contrast_criterion = nn.MSELoss(reduction='mean')
            l2 = contrast_criterion(pos_energy, neg_energy)
            contrast_loss = torch.pow(torch.clamp(threshold - l2, min=0.0), 2)
        else:
            raise NotImplementedError
        
        return contrast_loss.squeeze()

    def contrast_loss_revised(self, energy_dist, xu, threshold=2.0, mode='hinge'):
        """
        energy_dist: (B, V, U) energies
        xu: (B, full_len), zero-padded
        Returns scalar tensor.
        """
        B, V, U = energy_dist.shape
        # Positions of non-pad targets at this step: u -> (B, U)
        u = self.get_2D_indices(array=xu, val=self.tokenizer.pad_token_id, type='remove_pad')
        assert u.size(0) == B and u.size(1) == U
        # Gold token ids at those positions: (B, U)
        pos_ids = xu.gather(dim=1, index=u)
        # Positive energies per position: (B, U)
        pos_E = energy_dist.gather(dim=1, index=pos_ids.unsqueeze(1)).squeeze(1)
        # Hard negative per position = best competitor under current model.
        # Get two smallest energies/ids along V: (B, 2, U)
        vals, ids = torch.topk(energy_dist, k=2, dim=1, largest=False)
        top1_ids = ids[:, 0, :]   # best (lowest energy)
        top2_ids = ids[:, 1, :]   # second best
        # If the best is actually the gold id, take the second best; else take the best.
        neg_ids = torch.where(top1_ids.eq(pos_ids), top2_ids, top1_ids)  # (B, U)
        # Negative energies per position: (B, U)
        neg_E = energy_dist.gather(dim=1, index=neg_ids.unsqueeze(1)).squeeze(1)
        # Directional per-position loss
        if mode == 'hinge':
            # Enforce pos_E + margin <= neg_E
            per_pos = F.relu(threshold + pos_E - neg_E)
        elif mode == 'softplus':
            # Smooth logistic pairwise loss; small when pos_E << neg_E
            per_pos = F.softplus(pos_E - neg_E)
        else:
            raise ValueError("mode must be 'hinge' or 'softplus'.")
        return per_pos.mean()
    
    # a InfoNCE-style conrtast loss
    def contrast_loss_infonce(self, energy_dist, xu, k=5, temperature=1.0):
        """
        Sample top-k competing tokens (excluding gold) per position and apply InfoNCE.
        """
        B, V, U = energy_dist.shape
        u = self.get_2D_indices(array=xu, val=self.tokenizer.pad_token_id, type='remove_pad')
        pos_ids = xu.gather(dim=1, index=u)
        pos_E = energy_dist.gather(1, pos_ids.unsqueeze(1)).squeeze(1)  # (B, U)

        # Get top-(k+1) lowest-energy ids; drop gold if present, keep k negatives
        vals, ids = torch.topk(energy_dist, k=min(k+1, V), dim=1, largest=False)  # (B, k+1, U)
        # Build mask to remove gold
        gold_mask = ids.eq(pos_ids.unsqueeze(1))  # (B, k+1, U)
        # Replace any gold occurrence by a large energy so it’ll be dropped
        vals_masked = vals + gold_mask.float() * 1e6
        # Take the k smallest after masking: (B, k, U)
        neg_vals, _ = torch.topk(vals_masked, k=min(k, vals_masked.size(1)-1), dim=1, largest=False)
        neg_E = neg_vals  # (B, k, U)

        # Convert energies to logits (higher better) with temperature
        pos_logit = (-pos_E / temperature).unsqueeze(1)         # (B, 1, U)
        neg_logits = (-neg_E / temperature)                     # (B, k, U)
        logits = torch.cat([pos_logit, neg_logits], dim=1)      # (B, 1+k, U)

        # CE over the small (1+k)-way set; positive index = 0
        log_probs = logits.log_softmax(dim=1)
        loss = -log_probs[:, 0, :].mean()
        return loss     
       
    def fast_contrast_loss(self, xo, xu, loss_type='l2', threshold=2):
        '''
        Direct calculation of the contrast loss between the positive and negative energies.
        Negative samples have a random number of [1, |u|] tokens at random output positions 
        
        inputs:
            xo: Size(batch_size, full_len), input and previously unmasked output
            xu: Size(batch_size, full_len), current unmasking tokens from positve label
        returns:
            contrast_loss: Sizes(1), scalar contrast value, using L2 contrast by default.
        '''
        u = self.get_2D_indices(
            array=xu,
            val=self.tokenizer.pad_token_id,
            type='remove_pad',
        )
        xu_ = xu.clone()
        xu_[xu == self.tokenizer.unk_token_id] = self.tokenizer.pad_token_id #recover the PAD tokens once replaced by UNK
        pos_xou, neg_xou = xo.clone(), xo.clone()
        # effect of: pos_xou[u], neg_xou[u] = pos_xu[u], neg_xu[u]
        pos_u_val = xu_.gather(dim=1, index=u)
        neg_u_val = self.make_negative(pos_u_val)
        pos_xou.scatter_(dim=1, index=u, src=pos_u_val)
        neg_xou.scatter_(dim=1, index=u, src=neg_u_val)
        # print(f'Inside fast_contrast_loss, pos_xou({pos_xou.shape}): \n{pos_xou},\nneg_xou({neg_xou.shape}): \n{neg_xou}')
         
        pos_energy, neg_energy = self.energy(pos_xou), self.energy(neg_xou) #Size(batch_size)
        # print(f'pos_enerrgy({pos_energy.shape}): {pos_energy},\nneg_energy({neg_energy.shape}): {neg_energy}')
        if loss_type == 'l2':
            contrast_criterion = nn.MSELoss(reduction='mean')
            l2 = contrast_criterion(pos_energy, neg_energy)
            contrast_loss = torch.pow(torch.clamp(threshold - l2, min=0.0), 2)
        else:
            raise NotImplementedError
        
        return contrast_loss.squeeze()
    
    def log_gibbs_dist(self, energy_dist, energy_clip=True):
        '''Basically performs Eq.(0), energy clip is appplied by default. 
        Similar to softmax on Size(batch_size, vocab_size)'''
        if energy_clip:
            energy_dist = energy_dist - energy_dist.min() #perform global clipping so that all energies are between 0~1
        # partition = torch.sum(torch.exp(-1*energy_dist), dim=-1) #Size(batch_size, |u|)
        # expanded_partition = partition.unsqueeze(-1).expand_as(energy_dist)
        # normed_gibbs_dist = torch.exp(-1*energy_dist) / expanded_partition
        log_gibbs_dist = torch.log_softmax(-1*energy_dist, dim=-1)
        # TODO: NaN problem: nominator becomes full of zeros
        no_nan = (torch.isnan(log_gibbs_dist).any() == False)
        assert no_nan, f'log_gibbs_dist contains non-positive or nan values!' \
            f'\nclipped energy({energy_dist.shape}): \n{energy_dist}' \
            f'\nno_nan:{no_nan}'
        return log_gibbs_dist
    
    def draw_landscape(self, sample_batch, partial_pred):
        raise #TODO
    
    def sampling_revised(self, partial_pred:torch.Tensor, u:torch.Tensor, \
        sampler='gibbs', sampling_times=10, visualize=False, batch_id=None, get_logits=False):
        '''
        Revised faster version of sampling(), batchalized all unmasking positions and class values inside each time t.
        
        - partial_pred: Size(batch_size, full_len), input and previously unmasked tokens
        - data: Size(batch_size, full_len), the processed data dict specified to k
            
        - returns:
            - cand_pred: Size(batch_size, full_len), the updated partial_pred after T iaterations,
            - stat: Dict with statistical infos about losses and visualizations
        '''
        # torch.set_printoptions(threshold=float('inf')) #########for checking
        # 1. Get all the needed sizes and indices
        B, full_len = partial_pred.size()
        V, device = self.vocab_size, self.device
        U = u.size(-1)
        if get_logits:
            stat = {}
            pseudo_xu_logits = []
        else:
            stat = None
        # print(f'k={batch_id}, u({u.shape}): {u}')
        if sampler == 'gibbs':
            prev_pred  = partial_pred.to(device) #(B, full_len)
            # 2. Start T iterations of Gibbs sampling
            for t in range(sampling_times):
                # print(f'_____Start t={t}-th sampling_____')
                if visualize: 
                    # TODO
                    raise
                # 2.1 Calculate the energy of the previous prediction
                prev_energy = self.energy(prev_pred) #(B,)
                # 2. Generate a new Gibbs sample from the previous sample
                cand_pred = prev_pred.clone() #(B,full_len)
                # print(f'the initial cand_pred at t={t}, ({cand_pred.shape}): \n{cand_pred}')
                v_range = torch.arange(V, device=device).view(1,V,1).expand(B,V,1)
                for i in range(U): # vectorized on the v dimension
                    xui_base = cand_pred.unsqueeze(1).expand(B,V,full_len)
                    # if i == 2 and t == 0 and batch_id == 2:
                    #     print(f'i={i}, initial xui_base({xui_base.shape}): \n{xui_base}')
                    ui = u[:,i].view(B,1,1).expand(B,V,1)
                    xui_base = xui_base.scatter(dim=-1, index=ui, src=v_range) #(B, V, full_len)
                    flat = xui_base.reshape(-1, full_len) #(B*V,full_len)
                    # if i == 2 and t == 0 and batch_id == 2:
                    #     print(f'\ni={i}, scattered xui_base({xui_base.shape}): \n{xui_base}')
                    #     print(f'flat.shape: {flat.shape}')
                    ui_energy = self.energy(flat).view(B,V)
                    log_ui_dist = self.log_gibbs_dist(ui_energy)
                    if t == (sampling_times-1) and get_logits:
                        pseudo_xu_logits.append(log_ui_dist.unsqueeze(1)) #(B,1,V)
                    try:
                        ui_dist = torch.exp(log_ui_dist) #(B,V)
                    except:
                        raise RuntimeError(f'ui_energy: \n{ui_energy}\nlog_ui_dist:\n{log_ui_dist}\nui_dist:\n{ui_dist}')
                    x_ui = torch.multinomial(ui_dist, num_samples=1).view(B,1)
                    # x_ui = torch.argmax(ui_dist, dim=-1, keepdim=True) #(B,1) #for optimal inference (temp)
                    cand_pred = cand_pred.scatter(dim=-1, index=u[:,i].view(B,1), src=x_ui)
                    # if i == 2 and t == 0 and batch_id == 2:
                    #     print(f'after sampling at position i={i}, cand_pred becomes: \n{cand_pred}')
                # end of pos i iter
                cand_energy = self.energy(cand_pred)
                
                # 3. Modify the previous prediction if the sampling result has lower energy
                energy_drop_mask = (cand_energy < prev_energy).unsqueeze(1).expand(B,full_len)
                curr_pred = torch.where(energy_drop_mask, cand_pred, prev_pred)
                
                # if i == 2 and batch_id == 2:
                #     print(f'The final candidate at t={t}: \n{cand_pred}')
                #     print(f'the previous energy({prev_energy.shape}): \n{prev_energy}')
                #     print(f'the candidate energy({cand_energy.shape}): \n{cand_energy}')
                #     print(f'energy_drop_mask ({energy_drop_mask.shape}): {energy_drop_mask}')
                #     print(f'curr_pred at t={t} ({curr_pred}): \n{curr_pred}')
                
                # raise######test
            # end of time t iter
            if get_logits:
                pseudo_xu_logits = torch.cat(pseudo_xu_logits, dim=1).permute(0,2,1).contiguous() #(B,V,U)
                assert pseudo_xu_logits.size() == (B,V,U), f'pseudo_xu_logits.size: {pseudo_xu_logits.size}'
                stat = {'pseudo_xu_logits': pseudo_xu_logits}
                
        return curr_pred, stat
    
    def sampling(self, order_label:int, partial_pred:torch.Tensor, sample_batch:dict, \
        sampler='gibbs', sampling_times=10, visualize=False, batch_id=None):
        '''
        Batchalized sampling.
        For inference, the tokens previously unmasked are predicted again.
        
        partial_pred: Size(batch_size, full_len), input and previously unmasked output
        '''
        batch_size = partial_pred.size(0)
        stat = None
        if sampler == 'gibbs':
            prev_pred = partial_pred
            for t in range(sampling_times):
                print(f'___Start t={t}-th sampling...')
                cand_pred = prev_pred.clone()
                if visualize and order_label==3 and t == sampling_times-1:
                    raise
                    landscape = self.draw_landscape( #TODO: new version without latent
                        sample_batch,
                        cand_pred,
                    )
                prev_energy = self.energy(prev_pred)
                print(f'prev_pred({prev_pred.shape}): {prev_pred}, \nprev_energy({prev_energy.shape}): {prev_energy}')
                
                unmask_idx = self.get_2D_indices(
                    array=sample_batch['schedule_label'],
                    val=order_label,
                    type='curr_unmask', #full_unmask
                ) #Size(batch_size, |u_<=|)
                print(f'unmask_idx({unmask_idx.shape}): {unmask_idx}')
                for i in range(unmask_idx.size(-1)):
                    print(f'{i}/{unmask_idx.size(0)}position, cand_pred[0]: \n{cand_pred[0]}')
                    # calculate the gibbs disttribution of energy at each unmasking position
                    dist_i = []
                    for v in range(self.vocab_size):
                        # assign value v to all the samples in the batch
                        seq_iv = cand_pred.clone()
                        seq_iv[unmask_idx[:, i]] = v
                        energy_iv = self.energy(seq_iv) 
                        dist_i.append(energy_iv.view(batch_size, 1))
                    dist_i = torch.cat(dist_i, dim=-1) #energy_dist, Size(batch_size, vocab_size)
                    dist_i = torch.exp(self.log_gibbs_dist(dist_i))
                    cand_i = torch.multinomial(dist_i, num_samples=1).squeeze(-1)
                    print(f'dist_i.shape: {dist_i.shape}, cand_i.shape: {cand_i.shape}')
                    cand_pred[unmask_idx[:, i]] = cand_i
                cand_energy = self.energy(cand_pred)

                if cand_energy < prev_energy:
                    prev_pred = cand_pred
                    print(f'energy decreased from {prev_energy} to {cand_energy}, prev_pred updated to {prev_pred}')
                break###################test
            #end of t iter
        else:
            raise NotImplementedError(f'sampler: {sampler} is not defined!')
        
        return cand_pred, stat
                