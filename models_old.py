import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import os
import os.path as osp
import time
import json
sys.path.append('/home/user/shiqi/yichuan/EBM/ire_reasoning')
os.chdir('/home/user/shiqi/yichuan/EBM/ire_reasoning')
# print(f'The current working directory: {os.getcwd()}')
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutput

from utils import convert_time, VisualizeEBMs, check_grad


def swish(x): #当做一种mlp的activation就好：在linear与ReLU之间可调。beta=1的时候(i.e. this definition)叫SiLU (Sigmoid Linear Unit)
    return x * torch.sigmoid(x)

class SinusoidalPosEmb(nn.Module): #正弦嵌入（Attenion is All You Need提出）
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

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
'''
Customized toknenizer for binary tasks when using 'transformer' parameterization
'''
class BinaryTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self.vocab = {"0": 0, "1": 1, "[MASK]": 2}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(**kwargs)

        # Add special tokens if needed (e.g., masking)
        self.add_special_tokens({"mask_token": "[MASK]"})

        self.pad_token_id = self.vocab["[MASK]"]
        self.eos_token_id = self.vocab["[MASK]"]

    def _tokenize(self, text, **kwargs):
        return list(text)  # Split input into characters

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get("[MASK]"))  # Handle unknown tokens

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "[MASK]")

    def get_vocab(self):
        return self.vocab.copy()

    def save_vocabulary(self, save_directory, **kwargs):
        # Save vocabulary (optional)
        import os
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_path,)

'''
暂时无用
'''
class RestrictedCausalModel(nn.Module):
    '''
    Wrapper class of common HuggingFace models to restrict input and output to specific token IDs

    Args:
        - base_model_name_or_path (str)
        - token_ids (list): List of allowed token IDs (e.g. [0,1,2] for "0", "1", and "masked")
    '''
    def __init__(self, base_model_name_or_path, token_ids):
        super().__init__()
        self.allowed_token_ids = token_ids
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        self.logits_mask = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
        self.logits_mask[self.allowed_token_ids] = True
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        masked_logits = outputs.logits.masked_fill(~self.logits_mask.to(outputs.logits.device), -float('inf'))
        return CausalLMOutput(
            logits=masked_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, input_ids, gen_config, **kwargs):
        return self.model.generate(input_ids, gen_config, **kwargs)

class SequentialEBM:
    '''
    sequential EBM wrapper for common base models
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
        self.model, self.tokenizer = self._build_model(self.parameterization)
        self.generate_call_count = 0

    def _build_model(self, parameterization):
        if parameterization == 'mlp':
            return MLP(self.inp_len, self.out_len), None
        elif parameterization == 'transformer':
            '''
            Using popular LLMs as the base model for training sequential EBMs. 
            Modified to restrict the input and output vocabulary sizes.
            '''
            #1. load model and tokenzier
            org_model_path = 'meta-llama/Llama-3.1-8B-Instruct'
            # restricted_model = RestrictedCausalModel(org_model_path, self.num_classes)
            restricted_model = AutoModelForCausalLM.from_pretrained(org_model_path)
            # tokenizer = AutoTokenizer.from_pretrained(org_model_path)
            tokenizer = BinaryTokenizer()

            #2. resize embeddings and lm_head to binary vocab size (2)
            old_embedding = restricted_model.get_input_embeddings()
            new_embedding = torch.nn.Embedding(2, old_embedding.embedding_dim)
            restricted_model.set_input_embeddings(new_embedding)
            restricted_model.lm_head = torch.nn.Linear(
                old_embedding.embedding_dim, 2, bias=False
            )

            # modified gen_config is defined in self.generate()
            return restricted_model, tokenizer
        else:
            raise NotImplementedError

    def energy(self, k, i, v, y_one_hot, prior_dist):
        '''
        return EBM_k(x_i=v, x_{-i}=y_{-i}), batchalized

        Args:
            k (int): the k-th EBM
            i (int): the x_i position in range(k)
            v (int): the value for x_i
            y_one_hot (Size(batch_size, k, d)): the one-hot form of groundtruth sequence (partially masked)
            prior_dist (Size(batch_size, k, d)): the model's partially masked output logits
        Return:
            energy score 
        '''
        batch_size = prior_dist.size(0)
        energy = torch.zeros(batch_size)
        if self.parameterization == 'transformer': 
            for j in range(prior_dist.size(1)):
                if j == i:
                    energy += prior_dist[:, j, v]
                else:
                    y = torch.argmax(y_one_hot, dim=-1) #Size(batch_size, k)
                    y_j = y[:, j] #Size(batch_size)
                     #use y_j as indices to access the logits in prior_dist at j-th position
                    energy += prior_dist[:, j, :][torch.arange(batch_size), y_j] 
            return energy
        else:
            return NotImplementedError

    def generate(self, x, k, device='cuda'):
        '''
        normal model generation without explicit sampling

        params:
            x (Size(batch_size, inp_len, vocab_size)): partially masked input_ids
            k (int): k-th EBM
        return:
            logits (Size(batch_size, k, vocab_size)): perdicted prior dist from k-th EBM
        '''
        self.generate_call_count += 1
        if self.parameterization == 'transformer':
            '''
            needs to transform: tensor -> text -> generate() -> text -> tensor
            '''
            gen_config = GenerationConfig(
                max_new_tokens=k, #output size = k
                output_scores=True,
                return_dict_in_generate=True,
                # top_k=self.num_classes, #exclude special tokens #和do_sample互斥！
                do_sample=False,
                output_logits=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            x = torch.argmax(x, dim=-1) #non-one-hot, Size(batch_size, inp_len)
            input_texts = [''.join(str(token.item()) for token in row) for row in x]
            # if self.generate_call_count == 1:
            #     print(f'input_texts: {input_texts}')
            input_tokens = self.tokenizer(input_texts, return_tensors='pt')
            output_dict = self.model.generate(
                    input_tokens['input_ids'],
                    gen_config,
                    attention_mask=input_tokens['attention_mask'],
                    tokenizer=self.tokenizer,
                    return_dict_in_generate=True,
                )
            output_texts = self.tokenizer.batch_decode(output_dict.sequences, skip_special_tokens=True)

            # if self.generate_call_count == 5:
            #     print(f'5-th call, output_dict: \n {output_dict}')
            #     print(f'output_texts: {output_texts}')

            output_logits = output_dict.logits #tuple[k](tensor Size(batch_size, num_classes))
            output_logits = torch.stack(output_logits, dim=1)
            # if self.generate_call_count == 5:
            #     print(f'5-th call, after reshape, output_logits.shape: {output_logits.shape}') #Size(batch_size, k, num_classes)

            return output_logits
        else:
            raise NotImplementedError
        return None

            
    def train(self, data_batches, train_config, task_config):
        '''
        Train a sequence of EBMs together.
        Using pseudolikelihood to estimate each EBM (joint prior dist)
        Using cross entropy to calculate the loss between the y_i and prediction q(x_i | x_{-i})

        params:
            - data_batches (list(tensor.Size(batch_size, inp_len+out_len, num_classes))): list of data_batches, each batch is a torch tensor.
        '''
        #____________hyper_params_____________
        device='cuda'
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=train_config.lr)
        epochs = train_config.epochs
        batch_size = train_config.batch_size
        sample_num = task_config.train_size
        stat_path = f'./stats/train/{task_config.name}_{self.parameterization}.jsonl' #to store loss and ebms statistics
        ckpts_path =  f'./ebm_ckpts/{task_config.name}_{self.parameterization}.pth' #to store the model checkpoints
        #_____________________________________
        train_start = time.time()
        print(f"The first batch's shape: {data_batches[0].shape}") #Size(batch_size, inp_len+out_len)
        with open(stat_path, 'w') as statfile:
            for epoch in tqdm(range(epochs), desc='epoch'):
                for bi, data in enumerate(tqdm(data_batches, desc='batch')):
                    data = torch.nn.functional.one_hot(data.type(torch.long), self.num_classes) # bachalize
                    x = data[:, :self.inp_len, :]
                    # print(f'x.shape: {x.shape}')
                    if train_config.visualize:
                        visual_ebms = [] #list to store the visualized sequence of EBM imagees, [k][k, v]
                        losses = []
                    for k in tqdm(range(1, self.ebm_num+1), desc='ebms'):
                        logits = self.generate(x, k)
                        '''Set the logits to be able to back-prop, so that the model can be tuned'''
                        logits.requires_grad = True

                        y_one_hot = data[:, self.inp_len:self.inp_len+k, :] #Size(batch_size, k, class_num)
                        y = torch.argmax(y_one_hot, dim=-1) #Size(batch_size, k)
                        # print(f"{k}-th EBM: \ny.shape: {y.shape}\nlogits.shape: {logits.shape}")
                        total_loss = 0.0
                        if train_config.visualize:
                            visual_ebm = torch.zeros(k, self.num_classes) #batch-averaged image of energy landscape for k-th EBM
                        #estimate the joint energy via pseudolikelihood
                        for i in range(k): 
                            optimizer.zero_grad()
                            yi = y[:, i] #Size(batch_size, class_num) ?
                            q_xi = torch.zeros(batch_size, self.num_classes) #, requires_grad=True?
                            for v in range(self.num_classes):
                                energy = self.energy(k, i, v, y_one_hot, logits) # Size(batch_size)
                                if train_config.visualize:
                                    visual_ebm[i, v] = energy[0] #visualize the first batch only
                                q_xi[:, v] = energy
                            #calculate cross entropy for conditional x_i
                            loss = criterion(q_xi, yi) #the average cross entropy of the sample batch, nn.CrossEntropyLoss autometically performs LogSoftmax
                            # print(f'k: {k}, i: {i}, loss: {loss}, q_xi[0]: {q_xi[0]}, yi[0]: {yi[0]}')
                            total_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                            # check_grad(self.model) ############
                        #end of iter x_i
                        losses.append(round(total_loss, 4))
                        visual_ebms.append(visual_ebm.detach().numpy().tolist())
                        # break###############################
                    #end of ebm iter
                    stat = {
                        'batch_id': bi,
                        'loss': losses,
                        'ebms': visual_ebms,
                    }
                    statfile.write(json.dumps(stat)+'\n')
                    if bi % 5 == 0:
                    # if True:###########################
                        print(f'\nThe {bi}-th batch, stat: \n{stat}\n')
                    # break######################################
                #end of batch iter
                # break##################################
            #end of epoch iter
            print(f'\nFinished training.')
            state_dict = self.model.state_dict()
            torch.save(state_dict, ckpts_path)
            print(f'\nCheckpoints saved to {ckpts_path}')
        #end of stat write
        print(f'training stat saved to {stat_path}')

    def load_ckpts(self):
        if self.parameterization == 'transformer':
            ckpts_path = f'./ebm_ckpts/{self.task_config.name}_{self.parameterization}.pth'
            state_dict = torch.load(ckpts_path)
            self.model.load(state_dict)
        else:
            raise NotImplementedError
        print(f'Loaded checkpoints from {ckpts_path}')

    def sampling(self, sample_batch, sampler='gibbs', device='cuda', sampling_config=None, visual_log=None):
        '''
        Sampling on  the given sample batch, returns the prediction batch, the evaluation result and a viualization.

        params:
            - sample_batch (Size(batch_size, inp_len+out_len, num_classes)): one-hot data batch (x, y)
            - sampler (str): the sampling method, default: Gibbs
            - sampling_config: batch_size, steps, temperature, metrics, etc.
            - visual_log (dict): an initialized serializable dict

        returns:
            - pred_sequence (Size(batch_size, out_len)): non-one-hot predicted sequence
            - visual_log (VisualEBMs): a class with a filled dict containing the energy landscapes (num_vars x num_classes) and losses
        '''
        #___________configs_____________
        batch_size = sampling_config.batch_size #shouold be one!!
        steps_num = sampling_config.steps
        T = sampling_config.temperature #?
        time_step=visual_log.time_step
        #____________________________________
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            if sampler == 'gibbs':
                x = sample_batch[:, :self.inp_len, :]
                # randomly initialize a full output sequences
                pred_seq = torch.randint(low=0, high=self.num_classes, size=(batch_size, self.out_len))
                softmax = nn.Softmax(dim=1)
                for k in tqdm(range(1, self.ebm_num+1), desc='k-th ebm'):
                    logits = self.generate(x, k) #Size(batch_size, k, num_classes)
                    for t in tqdm(range(steps_num), desc='t-th time'):
                        y = torch.argmax(sample_batch[:, self.inp_len:self.inp_len+k, :], dim=-1) #just for calculating the loss

                        #build conditional distribution (logits)
                        energy_landscape = torch.zeros(k, self.num_classes) # first sample in each batch!
                        losses = torch.zeros(k)
                        for i in range(k):
                            pred_seq_onehot = torch.nn.functional.one_hot(pred_seq, self.num_classes)
                            yi = y[:, i]
                            #build conditional distribution
                            q_xi = torch.zeros(batch_size, self.num_classes) #conditional distribution of xi in logprob
                            for v in range(self.num_classes):
                                #TODO: energy function still not suit for other parameterizations
                                energy = self.energy(k, i, v, pred_seq_onehot, logits) #Size(batch_size), q(x_i=v | x_{-i}=y_{-i})
                                # print(f'\nq_xi ({q_xi.shape}): {q_xi}\nenergy ({energy.shape}): {energy}')
                                q_xi[:, v] = energy
                            energy_landscape[i, :] = q_xi[0, :] #take the first sample's energies along the classes per batch
                            losses[i] = criterion(q_xi, yi)
                            #sample
                            q_xi = softmax(q_xi) #normalized dist
                            sample = torch.multinomial(q_xi, num_samples=1).squeeze() #Size(batch_size)
                            # if t == 0 and k == 2 and i == 3:
                            #     print('\nt=0, k=2, i=3')
                            #     print(f'sample from q(x_3|...) is: {sample}')
                            #     print(f"one-hot prediction sequence's shape: {pred_seq_onehot.shape()}") #Size(batch_size, out_len, num_classes)
                            
                            #update prediction sequence
                            pred_seq[:, i] = sample

                        #end of x_i iter
                        # print(f'\n{k}-th EBM, {t+1}-step: pred_seq: {pred_seq}, energy_landscape: {energy_landscape}')
                        if t % time_step == 0:
                            visual_log.screenshot(k, t, energy_landscape, torch.mean(losses).item())
                    #end of step iter
                #end of ebm iter
                return pred_seq
            else:
                raise NotImplementedError

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
        


    def evaluate(self, val_data, store_stat=False, sampling_config=None, visualize_config=None):
        '''
        Evaluate on the validation set.

        params:
            - val_data (list(tensor.Size(batch_size, inp_len+out_len, num_classes))): list of one-hot data_batches, each batch is a torch tensor.
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