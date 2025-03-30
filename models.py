import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import os
import os.path as osp
import time
import random
import json
sys.path.append('/home/user/shiqi/yichuan/EBM/ire_reasoning')
os.chdir('/home/user/shiqi/yichuan/EBM/ire_reasoning')
# print(f'The current working directory: {os.getcwd()}')
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutput

from utils import convert_time, VisualizeEBMs, check_grad

inf = 1000000

def swish(x): #当做一种mlp的activation就好：在linear与ReLU之间可调。beta=1的时候(i.e. this definition)叫SiLU (Sigmoid Linear Unit)
    return x * torch.sigmoid(x)

def shuffle(index_list):
    snh48 = inde_list.copy()
    snh48 = random.shuffle(snh48)
    return snh48

'''Randomly change p-rate of the elements in a tensor within the value range'''
def random_flip(samples: Optional[torch.Tensor, list[torch.Tensor]], flip_range:int, rate=0.5)
    if isinstance(samples, torch.Tensor):
        samples = [samples]
    filpped_samples = []
    for sample in samples:
        flipped_sample = sample.clone()
        mask = torch.rand(sample.size()) < rate
        random_values = torch.randint(0, flip_range, sample.size(), dtype=torch.long)
        flipped_samples.append(torch.where(mask, random_values, flipped_sample))
        
    return flipped_samples

# Borrowed from IRED EBM class
class MLP(nn.Module):
    '''
    Modified from IRED (Simplified?).
    
    input: (inp_len+out_len)*voc_size
    output: out_len*class_num
    '''
    def __init__(self, inp_dim, out_dim, is_ebm=True):
        super(MLP, self).__init__()
        h = 512
        
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_ebm = is_ebm

        self.fc1 = nn.Linear(inp_dim + out_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim if is_ebm else out_dim)

    def forward(self, *args):
        if self.is_ebm:
            x = args
        else:
            x, y = args
            x = torch.cat([x, y], dim=-1)
            
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        h = swish(self.fc3(h))

        if self.is_ebm:
            output = self.fc4(h).pow(2).sum(dim=-1)[..., None] #最后一个维度上平方和，然后添加一个none在最后
        else:
            output = self.fc4(h) #raw prediction dist (not normalized)?

        return output
    
class SequentialEBM():
    '''
    sequential EBMs wrapper for different model architectures
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
        
    def _build_model(self, parameterization):
        if parameterization == 'mlp':
            return MLP(self.inp_len, self.out_len)
        else:
            raise NotImplementedError

    
    def mask(self, x: torch.Tensor, y: torch.Tensor, mask_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        randomly mask y, concatenate x with the masked y as observable x_o, return x_o and the unobservable x_u
        
        params: 
            x: Size(batch_size, inp_len)
            y: Size(batch_size, out_len)
            mask_idx: the index representing the <MASK> state
            
        returns:
            xo: Size(batch_size, inp_len+out_len), obervables, input to the base model
            xu: Size(batch_size, |u|), unobservables, output to be unmasked
        '''
        out_len = y.size(1)
        num_masks = random.randint(1, out_len)  # Randomly choose the number of masks
        mask_positions = random.sample(range(out_len), num_masks)  # Randomly sample index positions for masks
        mask_list = [1 if i in mask_positions else 0 for i in range(out_len)]
        masked_y = y * (1-mask_list) + mask_idx * mask_list
        xo = torch.cat([x, masked_y], dim=1)
        xu = y[:, mask_list]
        print(f'Inside self.mask(): test masked remains ({masked_y[:mask_list].shape}):" \
            f"\n{masked_y[:mask_list]} \nshould be of Size(batch_size, {num_masks}) with value {mask_idx}')###########
        
        return xo, xu
        
        
        
    
    def energy(self, dim:int, val: Optional[int, None], rest_idx: torch.Tensor, latent: torch.Tensor) \
        -> Optional[torch.float, torch.Tensor]: #目前唯一用torch.gather()的地方? 注意rest_idx需要提前unsqueeze(-1)变2D
        '''
        E(x_ui (=val) ; rest_idx)
        
        Given a specified dimension, an index table and a latent value source, 
        calculate the energy value for the specified index value on that dimension, 
        or all the energy values along that dimension.
        
        params:
            dim: the specified dimension "ui", also implies using the |ui|-th EBM with [x_u1 ~ x_ui] as input  
            val: the index value "x_ui". 
            rest_idx: a index value table, represented by a 2D tensor of Size(|[x_u1 ~ x_ui]|, 1), i.e. len(x_u'_{<=i})
                        placeholder for the ground_truth (dim-inclusive) for training,
                        or the current sample state for inference
            latent: the value source for energy calculation, represented by a 2D tensor of Size(|[x_u1 ~ x_ui]|, num_classes) 
            
        return:
                energyscalar, if specified val;
                energy vector (across all the classes), if val is not specified.  
        '''
        assert latent.dim() == rest_idx.dim()
        if val:
            try:
                energy = torch.gather(input=latent, dim=-1, index=rest_idx)
            except:
                raise ValueError(f"gather energy from input(latent) of {latent.size()}" \
                    f"with index(rest_idx) of {rest_idx.size()} is not defined!")
        else:
            # expand the column of the original rest_idx from Size(|ui'|, 1) to Size(|ui'|, num_classes), where
            # the original element on 'dim'-th row is replaced by arange(num_classes), 
            # the elements on other rows are copied for 'num_classes' of times
            expanded_idx = torch.zeros(rest_idx.size(0), self.num_classes)
            expanded_idx[dim] = torch.arange(self.num_classes) #enumerate all vals on ui
            for r in range(expanded_idx.size(0)):
                if r != dim:
                    expanded_idx[r].fill_(rest_idx[r].item()) #fill the original val on other positions
            try: 
                energy = torch.gather(input=latent, dim=1, index=expanded_idx)
            except:
                raise ValueError(f"gather energy from input(latent) of {latent.size()}" \
                    f"with index(rest_idx) of {rest_idx.size()} is not defined!")
                
        return torch.sum(energy, dim=0) #sum along all ui positions 

    '''Sample batch version'''
    def train(self, data_batches: list[tuple[torch.Tensor, torch.Tensor]], train_config:dict, task_config:dict):
        '''
        Train a sequence of EBMs together.

        params:
            - data_batches (list([tensor.Size(batch_size, inp_len), tensor.Size(batch_size, out_len)])): 
                list of data_batches, each batch is a torch tensor.
        '''
        #____________hyper_params_____________
        device='cuda'
        criterion = nn.CrossEntropyLoss()
        corrupt_func = random_flip
        optimizer = optim.Adam(self.model.parameters(), lr=train_config.lr)
        epochs = train_config.epochs
        batch_size = train_config.batch_size #100
        sample_num = task_config.train_size
        stat_path = f'./stats/train/{task_config.name}_{self.parameterization}.jsonl' #to store loss and ebms statistics
        ckpts_path =  f'./ebm_ckpts/{task_config.name}_{self.parameterization}.pth' #to store the model checkpoints
        early_stop_threshold = 0.5 #?
        #_____________________________________
        train_start = time.time()
        with open(stat_path, 'w') as statfile:
            for epoch in tqdm(range(epochs, desc='epoch')):
                for bi, (x,y) in enumerate(tqdm(data_batches, desc='batch')):
                    print(f'{bi}-th sample batch: \nx({x.shape}): {x}\ny({y.shape}): {y}')
                    '''1. Random mask and corrupt the sample'''
                    xo, xu = self.mask(x, y)
                    print(f'\n xu ({xu.shape}): {xu}, xo ({xo.shape}): {xo}')
                    neg_xu, neg_xo = corrupt_func(xu, xo) 
                    print(f'\n neg_xu: {neg_xu}, neg_xo: {neg_xo}')
                    '''2. Generating latent vectors for energy calculation'''
                    gamma = self.model.forward(xo, is_ebm=False) #return the latent vectors for each u_i given x_o
                    print(f'\nafter generation, gamma.shape: {gamma.shape}') #Size(batch_size, out_len, num_classes)
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
                #end of batch iter
                break#############test
            #end of epoch iter
        #end of statfile write
        
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
            
    def evaluate(self, data, sampling_config, visual_config):
        '''
        Evaluate on the validation set.

        params:
            - data (list(tensor.Size(batch_size, inp_len+out_len, num_classes))): list of one-hot data_batches, each batch is a torch tensor.
            - store_stat (bool): whether store the evaluation statistics locally
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
        visual_ebms = VisualizeEBMs(task_config, visualize_config, sampling_config)
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
                # print(f'before batchalize, data.shape: {data.shape}, data: \n{data}') #torch.Size([1, 22])
                data = torch.nn.functional.one_hot(data.type(torch.long), self.num_classes) # bachalize
                # print(f'after batchalize, data.shape: {data.shape}, data: \n{data}')
                pred_sequence = self.sampling(data, sampling_config=sampling_config, visual_log=visual_ebms)
                if store_stat:
                    y_batch = torch.argmax(data[:, self.inp_len:, :], dim=-1) #Size(batch_size, out_len), non-one-hot
                    batch_stat = self.apply_metrics(pred_sequence, y_batch, stat_dict)
                    batch_stat['x,y'] = torch.argmax(data, dim=-1).cpu().tolist()
                    if store_stat:
                        statfile.write(json.dumps(batch_stat)+'\n')
                
                # break############################
            #end of batch iter
            if store_stat:
                visualfile.write(json.dumps(visual_ebms.ebms_log)+'\n')
                for name in self.metrics:
                    if name == 'acc':
                        stat_dict[name]['he'] /= self.task_config.val_size
                        stat_dict[name]['se'] /= (self.task_config.val_size * self.out_len)
                    else:
                        raise NotImplementedError
                print(f'\nFinal Result:\n{stat_dict}')
                statfile.write('\nFinal Result:\n' + json.dumps(stat_dict))
        #end of stat write
        print(f'\nFinished evaluation. \nstatistics saved to {stat_path}')