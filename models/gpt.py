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

IGNORE_INDEX = -100

'''
model and datasets from GPT-from-scratch and pretrained HF GPT 

borrowed from diffusion-vs-ar paper
'''

def pad_sequence(seq, padding_value, cut_len):
    if len(seq) >= cut_len:
        seq = seq[:cut_len]
    else:
        seq = seq+[padding_value]*(cut_len-len(seq))
    return seq

class GPTDataset(Dataset):
    def __init__(self, data_pair, max_len, tokenizer, max_new_tokens=32, ignore_pad_for_loss=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = data_pair #list of (input_str, output_str)
        self.max_new_tokens = max_new_tokens
        self.ignore_pad_for_loss = ignore_pad_for_loss 
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, item):
        '''
        Similar to diffu_vs_ar preprocess_supervised_dataset()
        
        param: item: line index
        return: model_input: 
            - input_ids: src + SEP + MASK's(len(tgt)) + EOS (*edited)
            - attention_mask: 1's (len(input_ids))
            - labels: MASK's (len(src)) + SEP + tgt + EOS (*edited)
        
        
        '''
        # prepare model_input (tokenized)
        model_inputs = {}
        src, tgt = self.lines[item]
        src_ids, tgt_ids = self.tokenizer.encode(src) + [self.tokenizer.sep_token_id], \
            self.tokenizer.encode(tgt) + [self.tokenizer.eos_token_id]
        tgt_ids, src_ids = tgt_ids[:(self.max_len)], \
            src_ids[-(self.max_len-len(tgt_ids)):] #cutoff to max_len
        source_mask = [IGNORE_INDEX] * len(src_ids)   
        '''modified: masks in input and labels'''
        labels = source_mask + tgt_ids
        input_ids = src_ids + tgt_ids
        # src_mask_ids, tgt_mask_ids = [self.tokenizer.mask_token_id] * len(src_ids) \
        #     + [self.tokenizer.sep_token_id], [self.tokenizer.mask_token_ids] * len(tgt_ids) \
        #     + [self.tokenizer.eos_token_id]
        # input_ids, labels = src_ids + tgt_mask_ids, src_mask_ids + tgt_ids 
        
        model_inputs["input_ids"] = input_ids
        # model_inputs["attention_mask"].append([1] * len(src_ids)+ [0]*(len(input_ids)-len(src_ids)))
        model_inputs["attention_mask"] = [1] * len(input_ids)
        model_inputs["labels"] = labels
        
        # padding
        model_inputs['input_ids'] = pad_sequence(model_inputs['input_ids'], \
            padding_value=self.tokenizer.pad_token_id, cut_len=self.max_len)
        model_inputs['attention_mask'] = pad_sequence(model_inputs['attention_mask'], \
            padding_value=1, cut_len=self.max_len)
        if self.ignore_pad_for_loss:
            pad_value = IGNORE_INDEX
        else:
            pad_value = self.tokenizer.pad_token_id
        model_inputs['labels'] = pad_sequence(model_inputs['labels'], \
            padding_value=pad_value, cut_len=self.max_len) #ignore pad token for loss #modifed: IGNORE_INDEX
        # model_inputs['bert_label'] = pad_sequence(tgt_ids, padding_value=IGNORE_INDEX, \
        #     cut_len=self.max_new_tokens) #max_new_tokens
        
        model_inputs = {k: torch.tensor(v) for k, v in model_inputs.items()}
        
        return model_inputs