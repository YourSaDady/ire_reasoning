import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random as rand
import math
import itertools
import transformers
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
from pathlib import Path
os.chdir('/home/yichuan/HKU/EBM')
print(f'The current working directory: {os.getcwd()}')

'''
model and datasets from GPT-from-scratch and pretrained HF GPT 

borrowed from diffusion-vs-ar paper
'''

def pad_sequence(lists, padding_value, cut_len):
    new_lists = []
    for l in lists:
        if len(l) >= cut_len:
            new_lists.append(l[:cut_len])
        else:
            new_lists.append(l+[padding_value]*(cut_len-len(l)))
    return new_lists

class GPTDataset(Dataset):
    def __init__(self, data_pair, max_len, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = data_pair
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, item):
        '''
        param: item: line index
        return: model_input (dict), concatenated source and target
        '''
        # prepare model_input (tokenized)
        model_input = {}
        src, tgt = self.lines[item]
        src_ids, tgt_ids = self.tokenizer.encode(src), self.tokenizer.encode(tgt)
        tgt_ids, src_ids = tgt_ids[:(self.max_len-2)], \
            src_ids[-(self.max_len-2-len(tgt_ids)):] #cutoff to max_len
        input_ids = src_ids + [self.tokenizer.sep_token_id] + tgt_ids \
            + [self.tokenizer.eos_token_id]
        model_input['input_ids'] = input_ids #t1 + t2
        model_input['attention_mask'] = [1]*len(input_ids) 
        model_input['src_mask'] = [1]*(len(src_ids)+1)
        model_input["input_ids"] = pad_sequence(model_input["input_ids"], \
            padding_value=self.tokenizer.pad_token_id, cut_len=self.max_len)
        model_input["attention_mask"] = pad_sequence(model_input["attention_mask"], \
            padding_value=1, cut_len=self.max_len)
        model_input["src_mask"] = pad_sequence(model_input["src_mask"], \
            padding_value=0, cut_len=self.max_len)
        
        model_input = {k: torch.tensor(v) for k, v in model_input.items()}
        
        return model_input