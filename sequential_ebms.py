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
sys.path.append('/home/yichuan/HKU/EBM/ire_reasoning')
os.chdir('/home/yichuan/HKU/EBM/ire_reasoning')
os.environ['WANDB_API_KEY'] = '3c06642500f1527ecd0328870ff61d36b5c17193'
# os.environ['CUDA_LAUNCH_BLOCKING']=1
# os.environ['TORCH_USE_CUDA_DSA']=1
# print(f'The current working directory: {os.getcwd()}')
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutput

from utils import convert_time, VisualizeEBMs, check_grad
from typing import Optional, Union, Callable
import wandb

from models.mlp import MLP
from models.bert import DiscreteDiffusion

inf = 1000000

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

class MLPSequentialEBMs():
    '''
    sequential EBMs wrapper for MLP model architectures
    '''
    def __init__(self, parameterization='mlp', task_config=None, special_tokens=['<mask>']):
        self.task_config=task_config
        self.inp_len = task_config.inp_len
        self.out_len = task_config.out_len
        self.ebm_num = self.out_len #assume the number of EBMs equals the total number of variables
        self.num_classes = task_config.num_classes #exclude the special tokens, eg. <mask>
        self.metrics = task_config.metrics #list of metric names
        #build vocabulary dictionary (include special tokens)
        self.vocab = {v_id:str(v_id) for v_id in range(self.num_classes)}
        self.special_tokens = special_tokens
        for s_id in range(len(special_tokens)):
            self.vocab[self.num_classes+s_id] = special_tokens[s_id]

        self.parameterization = parameterization
        self.model = self._build_model(self.parameterization)
        self.generate_call_count = 0
        self.device= 'cpu' #'cuda'
        
    def _build_model(self, parameterization):
        if parameterization == 'mlp':
            return MLP(self.inp_len, self.out_len, self.num_classes)
        elif parameterization == 'bert':
            return DiscreteDiffusion(vocab_size=self.num_classes) #x.shape = Size(batch_size, 1, seq_len, seq_len)
        else:
            raise NotImplementedError
        

    def mask(self, x: torch.Tensor, y: torch.Tensor, mask_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        randomly mask y, concatenate x with the masked y as observable x_o.
        
        params: 
            x: Size(batch_size, inp_len)
            y: Size(batch_size, out_len)
            mask_idx: the index representing the <MASK> state
            
        returns:
            xo: Size(batch_size, inp_len+out_len), obervables, input to the base model
            xu: Size(batch_size, |u|), unobservables, output to be unmasked
            u: the list of masked indices
        '''
        out_len = y.size(1)
        num_masks = random.randint(1, out_len)  # Randomly choose the number of masks (normally dist)
        u = random.sample(range(out_len), num_masks)  # Randomly sample index positions for masks (already shuffled!)
        mask_list = [1 if i in u else 0 for i in range(out_len)] #all samples in a batch share the same masking indices
        mask = torch.tensor(mask_list)
        # print(f'Inside self.mask, randomly generated mask_list = {mask_list}')
        masked_y = y.clone()
        masked_y[:, mask==1] = mask_idx
        xo = torch.cat([x, masked_y], dim=1)
        xu = y[:, mask==1] #order maintaineds
        # print(f'test masked remains ({y[:, mask==1].shape}):' \
        #     f'\n{masked_y[:, mask==1]} \nshould be of Size(batch_size, {num_masks}) with value {mask_idx}')###########
        
        return xo, xu, u
        
        
    def energy(self, idx:int, val: bool, rest_idx: torch.Tensor, latent: torch.Tensor, device='cpu') \
        -> torch.Tensor: #目前唯一用torch.gather()的地方? 注意rest_idx需要提前unsqueeze(-1)变2D #cuda
        '''
        E(x_ui (=val) ; rest_idx)
        
        Given a specified dimension, an index table and a latent value source, 
        calculate the energy value for the specified index value on that dimension, 
        or all the energy values along that dimension.
        
        params:
            idx: the specified index "i" to predict
            val: whether the predict value "x_{u'_i}" is given, determines what to return
            rest_idx: the conditional values (include pos i!), represented by a 3D tensor of Size(batch_size, |x_{u'_{<=i}}|, 1)
            latent: the value source for energy calculation, represented by a 3D tensor of Size(batch_size, |x_{u'_{<=i}}|, num_classes) 
            
        return:
                energy: Size(batch_size), if specified val;
                energy vector: Size(batch_size, num_classes), if val is not specified.  
        '''
        assert latent.dim() == rest_idx.dim() and latent.size(1) == rest_idx.size(1), \
            f"latent.shape = {latent.shape}, rest_idx.shape = {rest_idx.shape}" #3, Size(batch_size, |u'i|, num_classes), Size(batch_size, |u'i|, 1)
        if val:
            energy = torch.gather(input=latent, dim=-1, index=rest_idx)
            # except:
            #     raise ValueError(f"gather energy from input(latent) of {latent.size()}" \
            #         f"with index(rest_idx) of {rest_idx.size()} is not defined!")
        else:
            # expand the last dim of the original rest_idx from Size(batch_size, |ui'|, 1) to Size(batch_size, |ui'|, num_classes), where
            # the original class value on ui position is replaced by arange(num_classes), 
            # the class values on positions are copied for 'num_classes' of times
            expanded_idx = torch.zeros(rest_idx.size(0), rest_idx.size(1), self.num_classes)
            for pos in range(expanded_idx.size(1)):
                if pos == idx: #enumerate all vals on ui
                    expanded_idx[:, pos, :] = torch.arange(self.num_classes) 
                else: #fill the original val on other positions for each class
                    expanded_idx[:, pos, :] = rest_idx[:, pos, :].expand(-1, self.num_classes) 
            expanded_idx = expanded_idx.to(torch.long).to(device)
            # print(f'\nbefore expand, rest_idx({rest_idx.shape}): \n{rest_idx}, \nafter expand, expanded_idx({expanded_idx.shape}: \n{expanded_idx})')
            energy = torch.gather(input=latent, dim=-1, index=expanded_idx)
            # except:
            #     raise ValueError(f"gather energy from input(latent) of {latent.size()}" \
            #         f"with index(rest_idx) of {expanded_idx.size()} is not defined!")
                
        return torch.sum(energy, dim=1) #sum along all ui positions

    
    def sampling(self, sample_batch, sampler='gibbs', device='cpu', sampling_config=None, visual_ebms=None): #cuda
        '''
        Sampling on  the given sample batch, returns the prediction batch, the evaluation result and a viualization.

        params:
            - sample_batch (Size(batch_size, inp_len+out_len)): one-hot data batch (x, y)
            - sampler (str): the sampling method, default: Gibbs
            - sampling_config: batch_size, steps, temperature, metrics, etc.
            - visual_ebms (VisualEBMs): an empty dic for storing EBM landscapes

        returns:
            - y_pred (Size(batch_size, out_len)): non-one-hot predicted sequence
            - ebms_log (VisualEBMs.ebms_log): a filled dict containing the energy landscapes (num_vars x num_classes) and losses for a sample batch
        '''
        #___________configs_____________
        batch_size = sampling_config.batch_size #should be one!!
        steps_num = sampling_config.steps
        T = sampling_config.temperature #?
        time_step=visual_ebms.time_step #1
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        #____________________________________
        self.model.to(device)
        sample_batch = sample_batch.to(device)
        with torch.no_grad():
            if sampler == 'gibbs':
                onehot_sample_batch = torch.nn.functional.one_hot(sample_batch.type(torch.long), self.num_classes) #one-hot transfer
                x = onehot_sample_batch[:, :self.inp_len, :]
                '''1. Randomly initialize 'y_pred' and sample an unmasking order 'pi'.'''
                y_pred = torch.randint(low=0, high=self.num_classes, size=(batch_size, self.out_len)).to(device) #not one-hot
                pi = torch.randperm(self.out_len) #unmasking order
                # print(f'\nInside sampling(), the randomly initialized y_pred({y_pred.shape}): \n{y_pred}' \
                #     f'\nunmasking order: {pi}')
                for k in range(1, self.ebm_num+1):
                    # print(f'\n\n{k}-th EBM: ')
                    yo = torch.unsqueeze(y_pred[:, pi[:k]], 2) #.to(device) #shuffled, Size(batch_size, |pi_k|, 1)
                    '''2. Gibbs sampling on yo'''
                    for t in range(steps_num): #TODO: log the energy landscapes and losses
                        # print(f'\nInside {t}-th step: ')
                        # y = sample_batch[:, self.inp_len:self.inp_len+k] #just for calculating the loss
                        masked_y = torch.full((batch_size, self.out_len), 2).to(device)
                        # print(f'\nmasked_y before: \n{masked_y}')
                        masked_y[:, pi[:k]] = yo.squeeze(2) #这一步并不会改变yo.shape (Size(batch_size, k, 1))
                        # print(f'\nmasked_y after: \n{masked_y}')
                        model_input = torch.cat([sample_batch[:, :self.inp_len], masked_y], dim=1).float().to(device) #Size(batch_size, inp_len+out_len)
                        gamma = self.model.forward(model_input, is_ebm=False) #Size(batch_size, out_len, num_classes)
                        yo_energy = self.energy(idx=pi[0], val=True, rest_idx=yo, latent=gamma[:, pi[:k], :]) #Size(batch_size, 1)
                        # print(f'\nyo_energy({yo_energy.shape}): \n{yo_energy}')
                        #build conditional distribution (logits)
                        energy_landscape = torch.zeros((k, self.num_classes), device=device) # first sample in each batch!
                        losses = torch.zeros(k).to(device)
                        yo_prime = yo.clone()
                        for i in range(k):
                            ei_dist = self.energy(idx=pi[i], val=False, rest_idx=yo, latent=gamma[:, pi[:k], :]) #Size(batch_size, num_classes)
                            z_oi = torch.sum(torch.exp(-1*ei_dist), dim=-1) #Size(batch_size)
                            expanded_zoi = z_oi.unsqueeze(1).expand_as(ei_dist) #Size(batch_size, num_classes)
                            # Sample from the 1D conditional p(y_{o_i} | y_{o_-i})
                            p_oi = torch.exp(-1*ei_dist) / expanded_zoi #Size(batch_size, num_classes)
                            
                            try:
                                y_oi_prime = torch.multinomial(p_oi, num_samples=1) #Size(batch_size, 1)
                            except:
                                raise RuntimeError(f'multinomial() input: p_oi seems contains nan.\n' \
                                    f'p_oi({p_oi.shape}): \n{p_oi}\n' \
                                    f'ei_dist({ei_dist.shape}): \n{ei_dist}\ngamma[:, pi[:k], :]({gamma[:, pi[:k], :].shape}): \n{gamma[:, pi[:k], :]}\n'\
                                    f'gamma({gamma.shape}): \n{gamma}\n'\
                                    f'model_input({model_input.shape}): \n{model_input}')
                            
                            # print(f'\nThe sampled p_oi.shape = {p_oi.shape}') #Size(1, 2)
                            yo_prime[:, i, :] = y_oi_prime
                            if t % time_step == 0:
                                # Record the i-th row of the energy landscape and losses at (k, t)
                                energy_landscape[i, :] = ei_dist[0, :] #Size(1, num_classes)
                                yi = sample_batch[0, self.inp_len+pi[i]].view(-1).to(torch.long)
                                # print(f'\n CE between p_oi: {p_oi[0:1, :]} and yi: {yi}')
                                losses[i] = criterion(p_oi[0:1, :], yi)
                        # end position iter
                        
                        # try:
                        yo_prime_energy = self.energy(idx=pi[0], val=True, rest_idx=yo_prime, latent=gamma[:, pi[:k], :]) #Size(batch_size, 1)
                        # except:   
                        #     raise RuntimeError(f'Error!: rest_idx(yo_prime).shape: {yo_prime.shape}, latent(gamma[:, pi[:k], :]).shape: {gamma[:, pi[:k], :].shape}')
                        
                        '''3. Check if the energy decreases after a single Gibbs step'''
                        mask = yo_prime_energy < yo_energy
                        expanded_mask = mask.unsqueeze(2).expand_as(yo) #Size(batch_size, |pi_k|, 1)
                        # print(f'\nyo_energy({yo_energy.shape}): \n{yo_energy}\nyo_prime_energy({yo_prime_energy.shape}): \n{yo_prime_energy}' \
                        #     f'\nexpanded_mask({expanded_mask.shape}): \n{expanded_mask}')
                        # print(f'\nBefore update, yo({yo.shape}): \n{yo}')
                        yo[expanded_mask] = yo_prime[expanded_mask] #update the entire sample row of yo if the energy decreases
                        # print(f'\nAfter update, yo({yo.shape}): \n{yo}')
                        if t % time_step == 0: 
                            # print(f'1st samples\' energy_landscape({energy_landscape.shape}): \n{energy_landscape}\nlosses({losses.shape}): {losses}')
                            visual_ebms.screenshot(k, t+1, energy_landscape, torch.mean(losses.cpu()).item())
                    #end of step iter
                    y_pred[:, pi[:k]] = yo.squeeze(-1) 
                    # print(f'predicted y = \n{y_pred}, ground_truth y = \n{sample_batch[:, self.inp_len:]} \nafter T = {steps_num} steps')
                #end of ebm iter
                if visual_ebms.visualize:
                    return y_pred, visual_ebms.ebms_log
                else:
                    return y_pred, None
            else:
                raise NotImplementedError
    
    '''Sample batch version'''
    def train(self, data_batches: list[tuple[torch.Tensor, torch.Tensor]], train_config:dict, task_config:dict, visual_config:dict):
        '''
        Train a sequence of EBMs together.

        params:
            - data_batches (list(tensor.Size(batch_size, inp_len+out_len))): list of data_batches, each batch is a torch tensor.
        '''
        #____________hyper_params_____________
        device= 'cpu' #'cuda'
        # criterion = F.cross_entropy 
        criterion = nn.CrossEntropyLoss().to(device)
        corrupt_func = random_flip # Callable
        optimizer = optim.Adam(self.model.parameters(), lr=train_config.lr)
        epochs = train_config.epochs
        batch_size = train_config.batch_size #100
        sample_num = task_config.train_size
        stat_path = f'./stats/train/{task_config.name}_{self.parameterization}.jsonl' #to store loss and ebms statistics
        ckpts_path =  f'./ebm_ckpts/{task_config.name}_{self.parameterization}.pth' #to store the model checkpoints
        early_stop_threshold = train_config.early_stop_threshold #0.5 (?)
        if train_config.wandb:
            wandb.login()
            run = wandb.init(
                project=f'EBM_train-{task_config.name}_{self.parameterization}',  # Specify your project
                config={                        # Track hyperparameters and metadata
                    "learning_rate": train_config.lr,
                    "epochs": epochs,
                },
            )
        #_____________________________________
        self.model.to(device)
        train_start = time.time()
        early_stop = False
        with open(stat_path, 'w') as statfile:
            print(f'Write to stat file at "{stat_path}" to store loss and EBMs statistics...')
            for epoch in tqdm(range(epochs), desc='epoch'):
                for bi, batch in enumerate(data_batches):
                    # visual_ebms = VisualizeEBMs(bi, task_config, visual_config) #sampling_config = None (steps=1)
                    x, y = batch[:, :self.inp_len], batch[:, self.inp_len:]
                    # print(f'{bi}-th sample batch: \nx({x.shape}): {x}\ny({y.shape}): {y}')
                    '''1. Random mask and corrupt the sample'''
                    xo, xu, u = self.mask(x, y, mask_idx=self.num_classes) #xo = x + y_masked, xu = masked y elements
                    # print(f'\nxu: \n{xu}, \nu: \n{u}')
                    neg_xu = corrupt_func(samples=xu, flip_range=self.num_classes) 
                    # print(f'\n neg_xu: {neg_xu}')
                    '''2. Generating latent vectors for energy calculation'''
                    xo = xo.float().to(device)
                    xu = xu.to(torch.long).to(device) #needed for indexing
                    gamma = self.model.forward(xo, is_ebm=False) #Size(batch_size, out_len, num_classes)
                    '''3. Pseudo-likelihood training for i = 1 to |u'| EBMs'''
                    # logp_xu, l_contrast = torch.zeros((batch_size, len(u), self.num_classes), requires_grad=True, device=device), \
                    logp_xu, l_contrast = [], \
                        torch.zeros(batch_size, requires_grad=True, device=device) #Initialize
                        
                    # energy_landscape = torch.zeros((len(u), self.num_classes), device=device) # first sample in each batch!
                    
                    for i in range(len(u)): #iter through |u'| number of EBMs (one single backprop path per iter)
                        i_order = torch.argsort(torch.tensor(u[:i+1]), dim=-1).tolist() #index within the u_{<=i} list
                        # print(f'\n\ni_order: {i_order}, ui = {u[:i+1]}')
                        condition_vals = torch.unsqueeze(xu[:, i_order], 2) # inclusive, the condition: x_{u'_{<=i}} # Size(bacth_size, |u'i|, 1)
                        # calculate the logp_xu increment (no order issues)
                        # print(f'Input data to the energy function: \nidx: {i}, \nrest_idx: \n{condition_vals}, \nlatent: \n{gamma[:, u[:i+1], :]}')
                        # Note that rest_idx and gamma are already shuffled and of |u'i| length
                        latent = gamma[:, u[:i+1], :]
                        if latent.dim() == 2: #usually when |u'i| = 1
                            latent = torch.unsqueeze(latent, 1)
                        pos_ei_dist = self.energy(idx=i, val=False, rest_idx=condition_vals, latent=latent)  #Size(batch_size, num_classes)
                        pos_ei = self.energy(idx=i, val=True, rest_idx=condition_vals, latent=latent)
                        z_ui = torch.sum(torch.exp(-1*pos_ei_dist), dim=-1) #Size(batch_size)
                        expanded_z_ui = z_ui.unsqueeze(1).expand_as(pos_ei_dist) #copy each row element by num_classes
                        logp_xui = torch.log(torch.exp(-1*pos_ei_dist) / expanded_z_ui)
                        # print(f'The {i}-th position, logp_xui({logp_xui.shape}) = \n{logp_xui}')
                        # logp_xu = logp_xu + torch.log(torch.exp(-1*pos_ei_dist) / expanded_z_ui) #Eq.(0), Size(batch_size, num_classes)
                        logp_xu.append(logp_xui.unsqueeze(1)) #Eq.(0), with sum replaced by concatenation
                        # energy_landscape[i, :] = pos_ei_dist[0, :]
                        # print(f'pos_ei_dist({pos_ei_dist.shape}): \n{pos_ei_dist}\nz_ui({z_ui.shape}): \n{z_ui}\nexpanded_z_ui(expanded_z_ui.shape): \n{expanded_z_ui}\nlogp_xu({logp_xu.shape}): \n{logp_xu}')
                        # calculate the contrast loss increment
                        neg_xu = torch.tensor(neg_xu, dtype=torch.long).to(device)
                        condition_vals = torch.unsqueeze(neg_xu[:, i_order], 2)
                        neg_ei = self.energy(idx=i, val=True, rest_idx=condition_vals, latent=latent) 
                        pos_ei, neg_ei = torch.squeeze(pos_ei), torch.squeeze(neg_ei)
                        # print(f'pos_ei({pos_ei.shape}): \n{pos_ei}\nneg_ei({neg_ei.shape}): \n{neg_ei}')
                        l_contrast = l_contrast - torch.log(torch.exp(-1*pos_ei) / (torch.exp(-1*pos_ei) + torch.exp(-1*neg_ei))) #Eq.(2), batch_size
                        # print(f'l_contrast: \n{l_contrast}')
                        # break ####################
                    #end of EBM iter
                    '''4. Calculate loss and backprop'''
                    logp_xu = torch.cat(logp_xu, dim=1)
                    p_xu = torch.exp(logp_xu)
                    # print(f'\n\nlogp_xu ({logp_xu.shape}): \n{logp_xu}')
                    # print(f'\n\np_xu ({p_xu.shape}): \n{p_xu}')
                    
                    logp_xu_flattened = logp_xu.view(-1, logp_xu.size(-1))
                    xu_flattened = xu.view(-1) # Size(batch_size, |u'|) -> Size(batch_size*|u'|)
                    l_ce = criterion(logp_xu_flattened, xu_flattened)
                    l_contrast = l_contrast.mean()
                    # print(f'l_ce: {l_ce}')
                    # print(f"l_contrast: {l_contrast}")
                
                    loss = l_ce + l_contrast ##scalar (batch-wise)
                    # print(f'\nloss = {loss}')
                    wandb.log({"l_ce": l_ce, "l_contrast": l_contrast, "loss": loss})
                    loss.backward()
                    optimizer.step()
                    
                    if loss < early_stop_threshold:
                        early_stop = True
                        break
                    
                    stat = {
                        'epoch': epoch,
                        'batch_id': bi,
                        'l_ce': l_ce.cpu().item(),
                        'l_contrast': l_contrast.cpu().item(),
                        'loss': loss.cpu().item(),
                        # 'ebms': visual_ebms,
                    }
                    statfile.write(json.dumps(stat)+'\n')
                        
                    # if bi == 10: #####################test
                    #     break#####################test
                #end of batch iter
                if early_stop:
                    break
                # break#############test
            #end of epoch iter
            state_dict = self.model.state_dict()
            torch.save(state_dict, ckpts_path)
            print(f'\nFinished training, Checkpoints saved to {ckpts_path}')
        #end of statfile write
        spent = convert_time(train_start)
        print(f'Training spent {spent[0]}:{spent[1]}:{spent[2]}.\n')
        
    '''Single data sample version'''
    def train_single(self, data, train_config, task_config):
        for sample_id, (x, y) in enumerate(data):
            print(f'{sample_id}-th sample: \nx({x.shape}): {x}\ny({y.shape}): {y}')
            '''1. Random masking and corrupting the unobservables'''
            u = self.random_mask(y.size(0)) 
            xu, xo = self.mask(x, y, u) #TODO: concatenate masked y and x as the model input: x_o, the golden label is x_u
            neg_xu, neg_xo = corrupt_func(xu, xo) #TODO: get corrupted xu and xo
            '''2. Generating latent vectors for energy calculation'''
            gamma = self.model.forward(xo, is_ebm=False) #return the latent vectors for each u_i given x_o
            '''3. Pseudo-likelihood training for i = 1 to |u'| EBMs'''
            u_prime = shuffle(u) # a list of shuffled unobserved indices
            logp_xu, l_contrast = torch.zeros(self.num_classes, requires_grad=True), torch.tensor(0.0, requires_grad=True) #Initialize
            for i, ui in enumerate(u_prime): #iter through |u'| number of EBMs (one single backprop path per iter)
                #TODO: 通过torch.gather和其他torch function实现求energy和q_ui
                # calculate the logp_xu increment
                pos_ei_dist = self.energy(dim=ui, val=None, rest_idx=xu, latent=gamma) #calculate the energies for all values along the given dim, #vector
                z_ui = torch.sum(torch.exp(-1*pos_ei_dist)) #scalar
                logp_xu += torch.log(torch.exp(-1*pos_ei_dist) / z_ui) #Eq.(0), vector
                # calculate the contrast loss increment
                pos_ei = pos_ei_dist[xu[i]]
                neg_ei = self.energy(dim=ui, val=neg_xu[i], rest_idx=neg_xu, latent=gamma)
                l_contrast -= torch.log(torch.exp(-1*pos_ei) / (torch.exp(-1*pos_ei) + torch.exp(-1*neg_ei))) #Eq.(2), scalar
            #end of EBM iter
            '''4. Calculate loss and backprop'''
            l_ce = criterion(logp_xu, xu) #scalar
            loss = l_ce + l_contrast
            loss.backward()
            optimizer.step()
                
            break#####################test
        #end of sample iter
            
    def apply_metrics(self, pred, y, stat_dict, task_config=None):
        '''
        Apply a list of names of evaluation metrics to a pair of (prediction, y), return in dict

        params:
            - pred (Size(batch_size, out_len)): prediciton sequence
            - y (Size(batch_size, out_len)): groundtruth
            - stat_dict (dict): the final statistics dict on the evaluation set
            - task_config: name, etc

        returns:
            - batch_stat (dict): batch statistics
        '''
        assert pred.shape == y.shape
        batch_size = pred.size(0)
        truth_matrix = (pred == y)
        batch_dict = {}
        for name in self.metrics:
            if name == 'acc': #here we only return the counts
                batch_dict[name] = {}
                acc_count = torch.sum(truth_matrix, dim=1) #sum along the second dimension, Size(batch_size)
                
                he_count = torch.sum(acc_count == self.out_len).item() #number of exactly mathcing sequences
                se_count = torch.sum(acc_count).item() #number of exactly matching variables
                stat_dict['acc']['he'] += he_count
                stat_dict['acc']['se'] += se_count
                batch_dict['acc']['he'] = he_count / batch_size
                batch_dict['acc']['se'] = se_count / (batch_size * self.out_len)
            else:
                raise NotImplementedError
        batch_dict['pred'] = pred.cpu().tolist()

        return batch_dict
    
    def evaluate(self, val_data, store_stat=False, sampling_config=None, visual_config=None):
        '''
        Evaluate on the validation set.

        params:
            - val_data (list(tensor.Size(batch_size, inp_len+out_len)))): list of x+y batches, each batch is a 2D tensor
            - store_stat (bool): whether store the evaluation statistics locally
            - sampling_config
            - visual_config
        '''
        #___________hyper params_____________
        path_prefix = f'./stats/evaluate/{self.task_config.name}_{self.parameterization}'
        visual_path = f'{path_prefix}_visual.jsonl'
        stat_path = f'{path_prefix}_stat.jsonl'
        
        task_config = {
            'num_ebms': self.ebm_num,
            'out_len': self.out_len,
            'num_classes': self.num_classes,
        }
        stat_dict = {}
        for name in self.metrics:
            if name == 'acc':
                stat_dict[name] = {}
                stat_dict[name]['he'] = 0
                stat_dict[name]['se'] = 0
            else:
                raise NotImplementedError
        #____________________________________
        with open(visual_path, 'w') as visualfile, open(stat_path, 'w') as statfile:
            for bi, data in enumerate(tqdm(val_data)):
                # print(f'\nInside evaluate(), the {bi}-th data.shape: {data.shape}, data: \n{data}') #torch.Size([3, 22])
                visual_ebms = VisualizeEBMs(bi, task_config, visual_config, sampling_config)
                pred_batch, ebms_log = self.sampling(data, sampling_config=sampling_config, visual_ebms=visual_ebms)
                # if store_stat:
                y_batch = data[:, self.inp_len:].to(self.device) #Size(batch_size, out_len)
                batch_stat = self.apply_metrics(pred_batch, y_batch, stat_dict)
                batch_stat['x,y'] = data.cpu().tolist()
                statfile.write(json.dumps(batch_stat)+'\n')
                if ebms_log:
                    visualfile.write(json.dumps(ebms_log)+'\n')
                # if bi == 10: ############################
                #     break############################
                # break ###################
            
            #end of batch iter
            # if store_stat:
            for name in self.metrics:
                if name == 'acc':
                    stat_dict[name]['he'] /= self.task_config.val_size
                    stat_dict[name]['se'] /= (self.task_config.val_size * self.out_len)
                else:
                    raise NotImplementedError
            print(f'\nFinal Result:\n{stat_dict}')
            if store_stat:
                statfile.write('\nFinal Result:\n' + json.dumps(stat_dict))
        #end of stat write
        print(f'\nFinished evaluation. \nstatistics saved to {stat_path}, \nvisualfile saved to: {visual_path}.')
        
        
        
class BERTSequentialEBMs():
    def __init__(self, task_config, d_model, device='cpu'):
        self.param_type = 'bert'
        self.task_config = task_config
        self.task_name = task_config.name
        self.inp_len = task_config.inp_len
        self.out_len = task_config.out_len
        self.max_len = self.inp_len + self.out_len + 3 #the max length model can take in
        self.vocab_size = task_config.num_classes
        self.special_tok_size = 4
        self.d_model = d_model
        self.device = device ###########for debugging
        self.criterion = nn.CrossEntropyLoss()
        
        self._build_model()
        
    def _build_model(self): #initialize a bert
        self.model = DiscreteDiffusion(
            vocab_size=self.vocab_size, 
            max_len=self.max_len,
            hidden_size=self.d_model,
        ) #Assume inp_len >= out_len
        
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
                
        return torch.sum(energy, dim=-2) #sum along all ui positions
        
    def gibbs_dist(self, energy_dist: torch.Tensor):
        '''
        Given the energy distribution at position i across all classes,
        calculate the Boltzmann (Gibbs) distribution. 
        
        params:
            energy_dist: Size(batch_size, num_classes)
        return:
            p_i: Size(batch_size, num_classes)
        '''
        z_i = torch.sum(torch.exp(-1*energy_dist), dim=-1) #Size(batch_size)
        expanded_zi = z_i.unsqueeze(1).expand_as(energy_dist) #Size(batch_size, num_classes)
        # Sample from the 1D conditional p(y_{o_i} | y_{o_-i})
        p_i = torch.exp(-1*energy_dist) / expanded_zi #Size(batch_size, num_classes)
        
        return p_i
        
    
    
    
    '''simplified(corrected?) non-batchalized version'''
    def sampling(self, order_label:int, partial_pred: torch.Tensor, sample_batch:dict, \
        sampler='gibbs', sampling_times=10, visual_ebms=None):
        '''
        Sampling on a partially masked sample batch. Gibbs sampling by default.
        
        params: 
            order_label: k (1-indexed)
            partial_pred: Size(bacth_size, seq_len)
            sample_batch: dict
            samping_times: T
            
        return:
            updated_partial_pred: Size(batch_size, seq_len)
            sth: loss_list or visual_ebms.log 
        '''
        # 1. Initialize yo and inputs to the model.forward and energy calcualtion
        batch_size = sample_batch['bert_input'].size(0)
        losses = []
        energies = []
        if sampler == 'gibbs':
            pred = []
            for b in range(batch_size):
                previous_pred = partial_pred[b]
                yo_idx = ((sample_batch['schedule_label'][b] > 0) & \
                    (sample_batch['schedule_label'][b] <= order_label)).nonzero(as_tuple=True)[0]
                # print(f'yo_idx.shape({yo_idx.shape}): \n{yo_idx}') #Size(|o|)
                yo = previous_pred[yo_idx]
                # print(f'yo({yo.shape}): \n{yo}')
                model_input = sample_batch['bert_input'][b].clone()
                model_input[model_input == 3] = 0 #replace the MASK with 0
                model_input += previous_pred
                # print(f"\nbert_input: {sample_batch['bert_input'][b]}\nprevious_pred: {previous_pred}"\
                #     f"\nmodel_input: {model_input}")
                for t in range(sampling_times):
                    # print(f'\n____\nStart t={t}-th sampling...\n')
                    gamma = self.model.forward(model_input.unsqueeze(0), \
                        sample_batch['segment_label'][b].unsqueeze(0), is_ebm=True).view(-1, self.vocab_size)
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
                        ei_dist = self.energy(idx=i, val=False, rest_idx=yo_prime.unsqueeze(-1), \
                            latent=gamma[yo_idx, :], batchalize=False)
                        p_oi = self.gibbs_dist(ei_dist.unsqueeze(0)) #Size(1,6)
                        
                        #_________forcing ignoring/considering the special tokens_______
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
                        # print(f'after update, previous_pred: {previous_pred}')
                        model_input = sample_batch['bert_input'][b].clone()
                        model_input[model_input == 3] = 0
                        model_input += previous_pred
                        # print(f'model_input: {model_input}')
                    # 4. Record partial prediction, losses(using logits) and energies (last sample in the batch)
                    if b == 0:
                        loss = self.criterion(gamma[yo_idx, :], sample_batch['bert_label'][b][yo_idx])
                        losses.append(round(loss.item(),2))
                        energies.append(round(yo_energy.item(),2))
                #end of t iter
                pred.append(previous_pred.view(1,-1))
                # break ####################test
            #end of inner-batch iter
        #end of 'gibbs'
        pred = torch.cat(pred, dim=0)
        # print(f'sampled pred[0] ({pred.shape}): {pred[0]}')
        
        return pred, {'losses': losses, 'energies': energies}
    
    
    '''Batchalized version of sampling (ineffective: single latent generation; repeated sampling input)'''
    def sampling_old(self, order_label:int, partial_pred: torch.Tensor, sample_batch:dict, \
        sampler='gibbs', sampling_times=10, visual_ebms=None):
        '''
        Sampling on a partially masked sample batch. Gibbs sampling by default.
        
        params: 
            order_label: k (1-indexed)
            partial_pred: Size(bacth_size, seq_len)
            sample_batch: dict
        return:
            updated_partial_pred: Size(batch_size, seq_len)
            sth: loss_list or visual_ebms.log 
        '''
        batch_size = sample_batch['bert_input'].size(0)
        model_input = sample_batch['bert_input'] + partial_pred
        model_input[model_input > self.vocab_size] -= 3 #subtract the MASK value (3), since added by the unmasked value
        # print(f"\nmodel_input: \n{model_input},\nprevious partial_pred: \n{partial_pred}")
        log_step = 1 #TODO
        if visual_ebms:
            log_step=visual_ebms.time_step #1
        loss_list = []
        criterion = nn.CrossEntropyLoss()
        if sampler == 'gibbs':
            '''1. generate latent'''
            with torch.no_grad():
                gamma = self.model.forward(model_input, sample_batch['segment_label'], is_ebm=True)
                # print(f'gamma.shape: {gamma.shape}') # Size(batch_size, seq_len, vocab_size) #includes special tokens
            # print(f'\norder_label: {order_label}')
            full_val = [] #x_ou, dim=2
            unmask_idx = [] # new x_u's indeices, dim=2
            full_idx = [] # x_ou's indices, dim=2
            for bid in range(batch_size):
                unmask = ((sample_batch['schedule_label'][bid] >=1) & \
                    (sample_batch['schedule_label'][bid] <= order_label)).nonzero(as_tuple=True)[0]
                full = (sample_batch['schedule_label'][bid] <= order_label).nonzero(as_tuple=True)[0]
                context = sample_batch['bert_input'][bid][full]
                if bid == 0:
                    yo_idx = (context == 3).nonzero(as_tuple=True)[0]#.tolist()
                full_val.append(context.view(1, -1))
                full_idx.append(full.view(1, -1))
                unmask_idx.append(unmask.view(1, -1))
            full_val = torch.cat(full_val, dim=0)
            full_idx = torch.cat(full_idx, dim=0)
            unmask_idx = torch.cat(unmask_idx, dim=0)
            #randomly initialize yo (exclide 4 special tokens, 4-indexed)
            yo = torch.randint(low=4, high=self.vocab_size, \
                size=unmask_idx.size()).to(self.device) #unmask_idx
            # print(f'\nbefore fillng yo, full_val: \n{full_val}\nyo_idx: ({yo_idx.shape})\n{yo_idx}')
            try:
                full_val[:, yo_idx] = yo #random initialize
            except:
                print(f"IndexError: sample_batch['bert_input']: \n{sample_batch['bert_input']}\nfull: \n{full}, context: \n{context},\nyo_idx: \n{yo_idx}, \nxu_idx: \n{xu_idx}")
            # print(f'\nafter fillng yo, full_val: \n{full_val}')
            full_val = torch.unsqueeze(full_val, -1)
            # print(f'full_idx({full_idx.shape}): \n{full_idx}')
            full_latent = torch.cat([gamma[r, full_idx[r], :].unsqueeze(0) for r in range(batch_size)],dim=0)
            # print(f'energy inputs: rest_idx({full_val.shape}), full_latent.shape: \n{full_latent.shape}')
            '''2. t steps of sampling'''
            for t in range(sampling_times): 
                yo_energy = self.energy(
                    idx=0, 
                    val=True, 
                    rest_idx=full_val,  #4,22,1
                    latent=full_latent, #4,22,6
                    batchalize=True    
                )
                yo_prime = yo.clone() #4,|u|
                if visual_ebms:
                    energy_landscape = torch.zeros((yo_prime.size(1), self.num_classes), \
                        device=self.device) # first sample in each batch!
                loss = torch.zeros(yo_prime.size(1))
                for i in range(yo_prime.size(1)): # iter through |u'|
                    ei_dist = self.energy( #Size(batch_size, num_classes) 4,6
                        idx=i,
                        val=False,
                        rest_idx=full_val,
                        latent=full_latent,
                        batchalize=True
                    ) 
                    p_oi = self.gibbs_dist(ei_dist) #Size(batch_size, num_classes) 4,6
                    try:
                        y_oi_prime = torch.multinomial(p_oi, num_samples=1) #Size(batch_size, 1) 4,1
                    except: #gradient explode / vanish
                        raise RuntimeError(f'multinomial() input: p_oi seems contains nan.\n' \
                            f'p_oi({p_oi.shape}): \n{p_oi}\n' \
                            f'ei_dist({ei_dist.shape}): \n{ei_dist}\ngamma[unmask_idx]'\
                            f'({gamma[unmask_idx].shape}): \n{gamma[unmask_idx]}\n'\
                            f'gamma({gamma.shape}): \n{gamma}\n'\
                            f'model_input({model_input.shape}): \n{model_input}')
                    # print(f'y_oi_prime({y_oi_prime.shape}): \n{y_oi_prime}')
                    # print(f'before update at i({i}), yo\': \n{yo_prime}')
                    yo_prime[:, i] = y_oi_prime.squeeze()
                    # print(f'after update at i({i}), yo\': \n{yo_prime}')
                    if t % log_step == 0:
                        if visual_ebms:
                            energy_landscape[i, :] = ei_dist[0, :]
                        yi = sample_batch['bert_label'][0, unmask_idx[0,i]].view(-1).to(torch.long)
                        loss[i] = criterion(p_oi[0:1, :], yi)
                #end of pos iter
                # print(f'\n\nafter all updates, yo\': \n{yo_prime}') #4,2
                full_val = full_val.squeeze(-1)
                # print(f'before update, full_val: \n{full_val}')
                full_val[:, yo_idx] = yo_prime
                # print(f'after update, full_val: \n{full_val}')
                full_val = full_val.unsqueeze(-1)
                yo_prime_energy = self.energy(
                    idx=0, 
                    val=True, 
                    rest_idx=full_val, 
                    latent=full_latent,
                    batchalize=True    
                ) #Size(batch_size, 1)
                '''Check if the energy decreases after a single Gibbs step'''
                mask = yo_prime_energy < yo_energy
                # print(f'yo_energy: \n{yo_energy}, \nyo_prime_energy: \n{yo_prime_energy}\nmask({mask.shape}): \n{mask}')
                expanded_mask = mask.expand_as(yo) #Size(batch_size, k, 1)
                yo[expanded_mask] = yo_prime[expanded_mask]
                if t % log_step == 0:
                    if visual_ebms:
                        visual_ebms.screenshot(
                            unmask_idx.size(1), 
                            t+1, 
                            energy_landscape,
                            torch.mean(losses.cpu()).item() 
                        )
                    else:
                        loss_list.append(round(torch.mean(loss).item(), 2))
            #end of t iter
            updated_partial_pred = partial_pred.clone()
            # print(f'\npartial_pred({partial_pred.shape}): \n{partial_pred}\nunmask_idx.shape: {unmask_idx.shape},\nyo.shape: {yo.shape}')
            for b in range(batch_size):
                updated_partial_pred[b][unmask_idx[b]] = yo[b]
            # print(f'after t-times sampling, updated_partial_pred: \n{updated_partial_pred}')
        #end gibbs
        else:
            raise NotImplementedError
        if visual_ebms:
            return updated_partial_pred, visual_ebms.ebm_log
        else:
            return updated_partial_pred, loss_list
        
    def calculate_contrast_loss(self, pos_latent, neg_latent, pos_label, neg_label, \
        pos_input, neg_input):
        '''
        Non-batchalized, calcualte the contrast loss based on Eq.(2), take sum from all classes
        '''
        # print(f'pos_input: \n{pos_input}, \npos_label:\n{pos_label}')
        # basically the same implementation as pseudolikelihood
        os_ = []
        u_primes = []
        pis = []
        for is_neg, (label, input) in enumerate(zip([pos_label, neg_label], \
            [pos_input, neg_input])):
            u = torch.nonzero(label).squeeze() #Size(|u|)
            o = torch.nonzero(input != 3).squeeze()
            if u.dim():
                pi = torch.randperm(u.size(0)) #a random estimating order
                u_prime = u[pi]
            else: # u is a single value
                pi = torch.tensor([0])
                u, u_prime = torch.tensor([u]), torch.tensor([u])
            if o.dim() == 0:
                o = torch.tensor([o])
            os_.append(o), u_primes.append(u_prime), pis.append(pi)
        # print(f'pos_input: \n{pos_input}, \nneg_input: \n{neg_input}')
        # print(f'pos_label: \n{pos_label}, \nneg_label: \n{neg_label}')
        # print(f'pos_pi: \n{pis[0]}, \nneg_pi: \n{pis[1]}')
        pos_eis, neg_eis = [], []
        pos_label, neg_label, pos_input, neg_input = pos_label.unsqueeze(-1), \
            neg_label.unsqueeze(-1), pos_input.unsqueeze(-1), neg_input.unsqueeze(-1)
        for i in range(len(u_primes[1])):
            for is_neg, (o, u_prime, pi, input, label, latent) in enumerate(zip(os_, u_primes, pis, \
                [pos_input, neg_input], [pos_label, neg_label], [pos_latent, neg_latent])):
                # print(f'\ninput({input.shape}): \n{input}, \no({o.shape}): \n{o}')
                full_vals = torch.cat([input[o, :], label[u_prime[:i+1], :]], dim=0)
                full_latent = torch.cat([latent[o, :], latent[u_prime[:i+1], :]], dim=0)
            
                ei = self.energy(idx=self.inp_len+2+pi[i], val=True, \
                    rest_idx=full_vals, latent=full_latent)
                if is_neg:
                    neg_eis.append(ei)
                else:
                    pos_eis.append(ei)
        assert len(pos_eis) != 0 and len(neg_eis) != 0, f'u_prime: {u_prime}\npos_label({pos_label.shape}): {pos_label}, \nneg_label({neg_label.shape}): {neg_label}'
        all_eis = torch.cat(pos_eis+neg_eis, dim=0) #Size(|u'i|)
        max_ei = all_eis.max()
        contrast_loss = torch.tensor(0., requires_grad=True).to(self.device)
        for pos_ei, neg_ei in zip(pos_eis, neg_eis):
            pos_ei, neg_ei = pos_ei-max_ei, neg_ei-max_ei
            contrast_loss = contrast_loss - torch.log(torch.exp(-1*pos_ei) / \
                (torch.exp(-1*pos_ei) + torch.exp(-1*neg_ei)))
        
        return contrast_loss

            

    
    def pseudolikelihood(self, latent, mlm_label, mlm_input):
        '''
        Non-batchalized! (single sample)
        
        estimte the logp using the model output logits and the MLM labels 
        params: 
            - latent: model output logits, Size(seq_len, num_classes)
            - mlm_label: input_ids for the masked tokens, Size(seq_len)
            - mlm_input: input_ids for the observed tokens, Size(seq_len), fully used their latents
        return:
            - lop_xu: the conditional logprob distribution, Size(seq_len, num_classes)
        '''
        assert latent.size(0) == mlm_label.size(0) == mlm_input.size(0), \
            f'latent.shape: {latent.shape}, ' \
            f'mlm_label({mlm_label.shape}): {mlm_label}, ' \
            f'mlm_input({mlm_input.shape}): {mlm_input}'
        # print(f'Inside pseudolikelihood: latent.shape: {latent.shape}, mlm_label.shape: {mlm_label.shape}')
        u = torch.nonzero(mlm_label).squeeze() #Size(|u|)
        o = torch.nonzero(mlm_input != 3).squeeze() #include MASK state (3)
        o_len = o.size(0)
        if u.dim():
            pi = torch.randperm(u.size(0)) #a random estimating order
            u_prime = u[pi]
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
        assert len(u_prime), f'u_prime is empty!! mlm_label: {mlm_label}, u: {u}, pi: {pi}'
        for i in range(len(u_prime)): #iter through |u'| EBMs
            condition_vals = mlm_label[u_prime[:i+1], :] #inclusive x_{u'_i} 121??
            condition_latent = latent[u_prime[:i+1], :]
            #note that MASK state is also included in the full_latent, the original tokens order is not maintained, but still matched
            #we found that including MASK state can give better performance? 
            full_vals = torch.cat([mlm_input[o, :], condition_vals], dim=0)
            full_latent = torch.cat([latent[o, :], condition_latent], dim=0)
            # print(f'\ni: {i}, rest_idx({condition_vals.shape}): \n{condition_vals}\ncondition_latent.shape: {condition_latent.shape}')
            
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
        logp_xu = torch.cat(logp_xu, dim=0) #Size(|u'|, num_classes)
        
        return logp_xu
    
    