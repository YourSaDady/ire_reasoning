import os.path as osp
import sys
import os
import json
import csv
import torch
import itertools
import random as rand
from torch.utils.data  import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
# sys.path.append('/root/EBM')
# os.chdir('/root/EBM')
# print(f'The current working directory: {os.getcwd()}')

from models.bert import train_tokenizer
from models.custom_tokenizer import CustomTokenizer
# from transformers import BertTokenizer

IGNORE_INDEX = -100

'''
Sudoku Task. Borrowed from IRED
'''
def get_data_dir(identifier):
    # base_dir = osp.join(osp.dirname(__file__), 'datasets') # ./datasets??
    base_dir = f'/home/user/shiqi/yichuan/EBM/datasets'
    print(f'base_dir: {base_dir}')
    if identifier.startswith('sudoku'):
        return osp.join(base_dir, 'sudoku')
    elif identifier.startswith('binary'):
        if identifier.endswith('addition'):
            return osp.join(base_dir, 'binary_arith/addition')
        elif identifier.endswith('subtraction'):
            return osp.join(base_dir, 'binary_arith/subtraction')
        elif identifier.endswith('inversion'):
            return osp.join(base_dir, 'binary_arith/inversion')
        else:
            raise ValueError('Unknown dataset: {}.'.format(identifier))
    else:
        raise ValueError('Unknown dataset: {}.'.format(identifier))

def load_satnet_dataset(data_dir): #废了
    if not osp.exists(data_dir):
        raise ValueError(f'Data directory {data_dir} does not exist. Run data/download-satnet.sh to download the dataset.')
    features = torch.load(osp.join(data_dir, 'features.pt'))
    labels = torch.load(osp.join(data_dir, 'labels.pt'))
    return features, labels

class SudokuDataset(Dataset): #TODO
    def __init__(self, data_pair, tokenizer, max_len=81): 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.corpus_line = len(data_pair)
        self.lines = data_pair
    
    def __len__(self):
        return self.corpus_line
    def __getitem__(self, item):
        '''
        return data_dict:
            - input (B, 81): zero-padded quizzes batch
            - label (B, 81): completed solution batch (the non-zero values in the inputs are maintained)
        '''
        # replace the given positions in the label with zero 
        q_list, s_list = [int(ch) for ch in self.lines[item][0]], [int(ch) for ch in self.lines[item][1]]
        q_tensor, s_tensor = torch.tensor(q_list), torch.tensor(s_list)-torch.tensor(q_list)
        # print(f'original quiz: \n{q_tensor.view(9,9)}\noriginal sol: \n{s_tensor.view(9,9)}')
        q_str, s_str = ''.join([str(ch) for ch in q_tensor.tolist()]), ''.join([str(ch) for ch in s_tensor.tolist()])
        quiz, sol = self.tokenizer.encode(q_str), self.tokenizer.encode(s_str)
        quiz = [self.tokenizer.mask_token_id if x == self.tokenizer.unk_token_id else x for x in quiz]
        sol = [self.tokenizer.pad_token_id if x == self.tokenizer.unk_token_id else x for x in sol]
        # print(f'encoded quiz: \n{torch.tensor(quiz).view(9,9)}\nencoded sol: \n{torch.tensor(sol).view(9,9)}')
        input_padding = [self.tokenizer.pad_token_id] * (self.max_len - len(quiz)) #[0,0,0,0], total = 85
        label_padding = [IGNORE_INDEX] * (self.max_len - len(sol)) #[-100, -100, -100, -100], total = 85
        quiz.extend(input_padding), sol.extend(label_padding)
        output = {
            'input': quiz,
            'label': sol
        }
        output = {k: torch.tensor(v) for k, v in output.items()}
        
        return output
        
        
class CountDownDataset(Dataset):
    def __init__(self, data_pair, tokenizer, stage, max_len=256, contrast=False):
        '''
        Init params:
            - data_pair: a list of (x, y) pairs, where x and y are strings, and words are separated by blank spaces
            - stage: pretrain / sft specify the masking area when calling __getitem__()
        '''
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair #  line: ('44,2,54,64', '2*54=108,108-44=64')
        self.stage = stage
        self.contrast = contrast
        
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, item): #"0", "1", MASK only, no segment label
        '''
        Select a positive sample pair from the data_pair, and preprocess.  
        
        params: item: line index
        return: output_dict:
            - input: masked context tensor of Size(max_len)
            - label: padded label tensor of Size(max_len)
        '''
        # print(f'\ninside _getitem_(), raw data: \n{self.lines[item]}')
        t2_all_zero = True
        while t2_all_zero: #randomly mask at least one position in t2
            # uniformly sample unmasking rate t
            t = rand.random()
            if t == 0:
                t = 0.05
            # get pos / neg sentence pair
            t1, t2 = self.get_sent(item, is_pos=True)
            # if self.contrast:
            #     neg_t2 = self.get_sent(item, is_pos=False) #same t1, but neg_t2
                
            # print(f'Inside __getitem__(), original t1: \n{t1}, \nt2: \n{t2}')
            # randomly mask t1 and t2 with a uniformly sampled masking rate t
            if self.stage == 'pretrain':
                (t1, t1_label), (t2, t2_label) = self.mask(t1, t), self.mask(t2, t) #tokenized
                t2_all_zero = all(x == 0 for x in t2_label)
            # mask t2 with random t while keep t1 unmasked
            elif self.stage == 'sft':
                (t1, t1_label), (t2, t2_label) = self.mask(t1, 0), self.mask(t2, t) #t
                t2_all_zero = all(x == 0 for x in t2_label)
            # keep t1 unmasked and t2 fully masked as the initial sequence 
            elif self.stage == 'inference': 
                # (t1, t1_label), (t2, t2_label) = self.mask(t1, 0), self.mask(t2, 1)
                (t1, t1_mask), (t2, t2_mask) = self.encode(t1), self.encode(t2) 
                # if self.contrast:
                #     neg_t2, neg_t2_mask = self.encode(neg_t2)
                #     longer = len(neg_t2) - len(t2)
                #     if longer>=0:
                #         neg_t2 = neg_t2[:len(t2)]
                #     else:
                #         neg_t2 = neg_t2 + [neg_t2[-1]]*(-1*longer)
                #     assert len(neg_t2) == len(t2), f'len(neg_t2): {len(neg_t2)}, len(t2): {len(t2)}'
                t2_all_zero = False
            else:
                raise NotImplementedError
        
        # print(f'After added special tokens, t1({len(t1)}): \n{t1}, \nt1_label: \n{t1_label}, \nt2({len(t2)}): {t2}, \nt2_label: {t2_label}')
        # concatenate t1 and t2 and add padding to max_len
        # print(f'\nself.max_len: {self.max_len}')
        input = (t1 + [self.tokenizer.sep_token_id] + t2_mask + \
            [self.tokenizer.eos_token_id])[:self.max_len]
        label = (t1_mask + [self.tokenizer.sep_token_id] + t2 \
            + [self.tokenizer.eos_token_id])[:self.max_len] #train: ignore src; inference: mask src
        attn = [1] * len(input)
        assert len(input) == len(label)
        input_padding = [self.tokenizer.pad_token_id] * (self.max_len - len(input))
        label_padding = [IGNORE_INDEX] * (self.max_len - len(input)) #self.tokenizer.pad_token_id
        attn_padding = [0] * (self.max_len - len(input))
        input.extend(input_padding), label.extend(label_padding), attn.extend(attn_padding)
        # if self.contrast:
        #     neg_label = (t1_mask + [self.tokenizer.sep_token_id] + neg_t2 \
        #     + [self.tokenizer.eos_token_id])[:self.max_len]
        #     neg_label.extend(label_padding)
        
        output = {
            'input': input,
            'label': label,
            'attention': attn,
        }
        # if self.contrast:
        #     output['neg_label'] = neg_label
        # print(f'final:\nraw line[idx]: \n{self.lines[item]}\nbert_input: \n{bert_input}, \nbert_label: \n{bert_label}') # \nsegment_label: \n{segment_label}
        output = {k: torch.tensor(v) for k, v in output.items()}
        # output['contrast'] = self.contrast
        
        return output
    
    def get_sent(self, idx, is_pos):
        if is_pos:
            t1, t2 = self.lines[idx][0], self.lines[idx][1]
            return t1, t2
        else: #TODO: method 1 and 2 are "bad" negatiuve samples
            raise NotImplementedError('you should not reach here')
            '''method1: random pair (meaningful negative target)'''
            # while True: #ensure the t2 is negative
            #     t2 = self.lines[rand.randrange(len(self.lines))][1]
            #     if t2 != self.lines[idx][1]:
            #         break
            '''method2: random token_id (meaningless negative target)'''
            # t2_len = len(self.encode(self.lines[idx][1])[0])
            # t2_ids = [rand.randint(5,20) for _ in range(t2_len)]
            # t2 = self.tokenizer.decode(t2_ids)
            '''method3: randomly select arbitrary number of tokens to shift by 1'''
            t2_ids = torch.tensor(self.encode(self.lines[idx][1])[0])
            # print(f'before, t2_ids: {t2_ids}, t2: {self.lines[idx][1]}')
            t2_len = len(t2_ids)
            shift_num = rand.randint(1, t2_len)
            idx = torch.randperm(t2_len)[:shift_num]
            delta = torch.randint(0, 2, (shift_num,))*2 - 1 #+1or-1
            t2_ids[idx] += delta
            t2 = self.tokenizer.decode(t2_ids)
            # print(f'after adding noise to positions: {idx}, t2_ids: {t2_ids}, t2: {t2}')
            
            # # print(f'neg t2: {t2}')
            return t2 
    
    def mask(self, sentence, t):
        tokens = sentence.split() #only useful for textual sentences
        output_label = []
        output =[]
        # t% of the tokens will be masked
        for i, token in enumerate(tokens): # iter through words
            prob = rand.random()
            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1] # token list
            # mask tokens of a word with t%
            if prob < t:
                for i in range(len(token_id)):
                    output.append(self.tokenizer._vocab_str_to_int["[MASK]"])
                output_label.append(token_id)
            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)
        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label
    
    def encode(self, sentence):
        sen_ids = self.tokenizer.encode(sentence)
        sen_mask = [self.tokenizer._vocab_str_to_int["[MASK]"]]*len(sen_ids)
        sen_ignore = [IGNORE_INDEX] * len(sen_ids)
        # print(f'inside encode(), t: {t}, \nsen_ids: \n{sen_ids}, \nsen_mask: \n{sen_mask}')
        return sen_ids, sen_mask

def _rescale(x): # 1 -> 1; 0 -> -1 (暂时不用)
    return (x - 0.5) * 2

def _decimal(tensor): #Sudoku: Size(9,9,9) -> Size(9,9) with observed values 1~9 and unobserved values 10
    assert tensor.size() == (9,9,9), "The tensor is not a one-hot feature for Sudoku!"
    mask = torch.sum(tensor, dim=-1) == 0
    decimal_sudoku = torch.argmax(tensor, dim=-1) + 1
    decimal_sudoku = torch.where(mask, torch.tensor(10), decimal_sudoku)
    return decimal_sudoku.view(9,9)

'''
Contains tasks: sudoku, binary-{addition, substraction, inversion}
'''
class SATDataset(Dataset): #identifier 就是 'binary_arith' + {split}
    def __init__(self, dataset_identifier, split):
        self.task = dataset_identifier
        self.features, self.labels = load_satnet_dataset(get_data_dir(dataset_identifier))
        nr_datapoints = len(self.features) #number of datapoints
        assert split in ('train', 'val')
        self.split = split
        if self.split == 'train':
            self.features = self.features[:int(nr_datapoints * 0.9)]
            self.labels = self.labels[:int(nr_datapoints * 0.9)]
            print(f'\nInside SATDataset __init__():\nself.features[0]({self.features[0].shape}): \n{self.features[0]}\n'\
                f'self.labels[0]({self.labels[0]}): \n{self.labels[0]}') # Size([9,9,9]), last row is one-hot vector (全0是masked)
        else:
            self.features = self.features[int(nr_datapoints * 0.9):]
            self.labels = self.labels[int(nr_datapoints * 0.9):]

        if dataset_identifier == 'sudoku':
            self.cond_entry = (self.features.sum(axis=-1) == 1)[:, :, :, None].expand(-1, -1, -1, 9) #condition entries
            print(f'\nself.cond_entry[0]({self.cond_entry[0].shape}): \n{self.cond_entry[0]}') # torch.Size([9, 9, 9]), 单行全为True / False
        self.inp_dim = self.features[0].numel()
        self.out_dim = self.labels[0].numel()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.task == 'sudoku':
            #return 3 x Size(9*9) with values [1~10], [1~9], [T/F]
            return _decimal(self.features[idx]).reshape(-1), _decimal(self.labels[idx]).reshape(-1), self.cond_entry[idx][:, :, 0].reshape(-1)
            # return _rescale(self.features[idx].reshape(-1)), _rescale(self.labels[idx].reshape(-1)), self.cond_entry[idx].reshape(-1)
        elif self.task.startswith('binary'):
            return self.features[idx], self.labels[idx]

'''sample_num * [Size(inp_len), Size(out_len)] -> batch_num * Size(batch_size, inp_len+out_len)'''
def batchlize(task, dataset, batch_size=1000): #TODO: add Sudoku specification
    batches = [] #list[Size(batch_size, inp_len+out_len)]
    if task.startswith('binary'):
        dataset = [torch.cat([data[0].unsqueeze(0), data[1].unsqueeze(0)], dim=1) for data in dataset] #list[Size(1, inp_len+out_len)]
        for idx in range(0, len(dataset), batch_size):
            if idx+batch_size < len(dataset): #ensure all batches are full with samples
                batches.append(torch.cat(dataset[idx:idx+batch_size], dim=0))
                assert batches[-1].size(0) == batch_size
    elif task == 'sudoku': # cannot batchalize (since masked positions differs among samples)
        batches = [
                    (
                        torch.cat([data[0].unsqueeze(0), data[1].unsqueeze(0)], dim=1), \
                        data[2] \
                    ) \
                        for data in dataset \
                ] #list[(feature_label: Size(1, 9*9+9*9), cond_entry: Size(9*9))]
    return batches

#######################################################################################################

def load_data_old(task, train_batch_size, val_batch_size):
    if task == 'sudoku' or task.startswith('binary'):
        train_set = SATDataset(task, split='train')
        val_set = SATDataset(task, split='val')
        print(f'train_set.inp_dim: {train_set.inp_dim}') #12 #729(sudoku)
        print(f'train_set.out_dim: {train_set.out_dim}') #10 #729(sudoku)
        print(f'first sample: {train_set[0]}') #(tensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=torch.int8), tensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=torch.int8))
        train_batches, val_batches = batchlize(task, train_set, train_batch_size), batchlize(task, val_set, val_batch_size)
        train_size, val_size = len(train_batches)*train_batch_size, len(val_batches)*val_batch_size
        print(f'train_size: {train_size}, val_size: {val_size}') #1800, 200 #8900, 999(sudoku)
        return train_batches, val_batches, train_size, val_size
    else:
        raise NotImplementedError(f'The specified task: {task} is not defined')
    
def load_gpt_data(task, max_len, train_batch_size, test_batch_size):
    # load tokenzier
    print(f'Now loading gpt tokenzier...')
    tokenizer = CustomTokenizer.from_pretrained('./ire_reasoning/models/model_config_tiny') 
    # reformat data into dataloader
    print(f'Now start loading train and test data...')
    if task == 'countdown':
        train_pairs, test_pairs = [], []
        for split, pairs in zip(['train', 'test'], [train_pairs, test_pairs]):
            path = f'./datasets/diffusion_vs_ar-data/cd3_{split}.jsonl' #pwd = EBM
            with open(path, 'r') as file:
                for line in file:
                    entry = json.loads(line.strip())
                    pairs.append((entry['input'], entry['output']))
            print(f'\n{split} set size: {len(pairs)}')
        # additional negative samples for the train set
        train_data, test_data = CountDownDataset(train_pairs, max_len=max_len, tokenizer=tokenizer, is_pos=False), \
            CountDownDataset(test_pairs, max_len=max_len, tokenizer=tokenizer) #max_new_tokens default = 32
        train_loader, test_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True), \
            DataLoader(test_data, batch_size=test_batch_size, shuffle=True, pin_memory=True)
    else:
        raise NotImplementedError
    print(f'train_data[0]: \n{train_data[0]}')
    return train_loader, test_loader, len(train_data), len(test_data)

def load_data(task, stage, max_len, train_batch_size, val_batch_size, contrast=False, parallel=False):
    test_count = 0
    '''
    params:
        stage: 
            - inference (default, fully masked target)
            - sft (partially masked target), 
            - pretrain (partially masked source and target sequences)
        max_len: the full length the model can take
        train_batch_size
        val_batch_size
        
    return:
        train_loader, test_loader, train_size, test_size, tokenizer
    '''
    # tokenizer = Tokenizer.from_file('./ire_reasoning/models/tokenizer_{task}.json')
    print(f'\nNow start loading train and test data...')
    if task.startswith('binary'):
        tokenizer_path = f'./models/tokenizer_{task}-vocab.txt'
        if osp.exists(tokenizer_path):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            print(f'\nFound tokenzier at {tokenizer_path}, loaded.')
        else:
            print(f'No existing tokenzier found, now start training tokenizer...')
            tokenizer = train_tokenizer(task)
        if task.endswith('addition'):
            # print(f'\nInside load_bert_data, the current working directory: {os.getcwd()}\n')
            train_path, test_path = '../datasets/binary_arith_txt/addition/train.txt', '../datasets/binary_arith_txt/addition/val.txt'
            train_pairs, test_pairs = [], []
            for path, pairs in zip([train_path, test_path], [train_pairs, test_pairs]):
                if path.endswith('train.txt'):
                    tag = 'train'
                else:
                    tag = 'test'
                with open(path, 'r') as file:
                    for line in file:
                        test_count += 1
                        # if test_count == 100: ########################
                        #     test_count = 0
                        #     break ########################
                        t1, t2 = line.strip().split(' +++$+++ ')
                        # t1, t2 = t1.split(), t2.split()
                        pairs.append((t1, t2))
                
                print(f'\n{tag} set size: {len(pairs)}')
        else:
            raise NotImplementedError        
        
        return train_loader, test_loader, len(train_data), len(test_data)
    elif task == 'countdown':
        # 1. load tokenizer
        print(f'pwd: {os.getcwd()}')
        tokenizer = CustomTokenizer.from_pretrained('./models/model_config_tiny/tokenizer_config_cd.json') 
        # 2. laod dataset
        train_pairs, test_pairs = [], []
        for split, pairs in zip(['train', 'test'], [train_pairs, test_pairs]):
            path = f'../datasets/cd/cd3_{split}.jsonl' #pwd = EBM
            with open(path, 'r') as file:
                for line in file:
                    entry = json.loads(line.strip())
                    pairs.append((entry['input'], entry['output']))
            print(f'\n{split} set size: {len(pairs)}')
        train_data = CountDownDataset(train_pairs, stage=stage, max_len=max_len, tokenizer=tokenizer, contrast=contrast) #TODO: contrast这个argument废了
        test_data = CountDownDataset(test_pairs, stage=stage, max_len=max_len, tokenizer=tokenizer)
    elif task == 'sudoku':
        tokenizer = CustomTokenizer.from_pretrained('./models/model_config_tiny/tokenizer_config_sudoku.json')
        train_pairs, test_pairs = [], []
        for split, pairs in zip(['train','test'], [train_pairs, test_pairs]):
            path = f'../datasets/sudoku/sudoku_{split}.csv'
            with open(path, newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader, None) #skip header
                for row in reader:
                    quiz, sol = map(str.strip, row)
                    pairs.append((quiz, sol))
            print(f'\n{split} set size: {len(pairs)}')
        train_data = SudokuDataset(train_pairs, max_len=max_len, tokenizer=tokenizer)
        test_data = SudokuDataset(test_pairs, max_len=max_len, tokenizer=tokenizer)
                        
                
    else:
        raise NotImplementedError
    
    print(f'train_data[0]: \n{train_data[0]}')
    
    if not parallel:
        train_loader, test_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True), \
            DataLoader(test_data, batch_size=val_batch_size, shuffle=True, pin_memory=True)
        return train_loader, test_loader, len(train_pairs), len(test_pairs), tokenizer
    
    else: #effective batch size is 32 * nprocs
        train_loader, test_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, \
            sampler=DistributedSampler(train_data)), DataLoader(test_data, batch_size=val_batch_size, \
            shuffle=False, sampler=None) #sampler=DistributedSampler(test_data) 
        return train_loader, test_loader, len(train_pairs), len(test_pairs), tokenizer