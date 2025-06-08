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

class GPTDataset(Dataset):
    '''Same to BERTDataset'''
    def __init__(self, data_pair, max_len, tokenizer, stage='inference', max_new_tokens=32, \
        ignore_pad_for_loss=True, is_pos=True, use_padding=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = data_pair #list of (input_str, output_str)
        self.max_new_tokens = max_new_tokens
        self.ignore_pad_for_loss = ignore_pad_for_loss 
        self.stage = stage
        self.is_pos = is_pos
        self.use_padding = use_padding
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, item):
        '''
        Similar to diffu_vs_ar preprocess_supervised_dataset()
        
        param: item: line index
        return: model_input: 
            - gpt_input: src + [SEP] + masked_tgt + [EOS] + [PAD*]
            - gpt_attn: 1's (len(input_ids))
            - gpt_label: masked_src + [SEP] + tgt + [EOS] + [PAD*]
        
        
        '''
        # prepare model_input (tokenized)
        src, tgt = self.get_sent()
        if self.stage == 'inference': 
            (src, src_mask), (tgt, tgt_mask) = self.encode(src), self.encode(tgt) 
        else:
            raise NotImplementedError
        
        gpt_input = (src + [self.tokenizer.sep_token_id] + tgt_mask + \
            [self.tokenizer.eos_token_id])[:self.max_len]
        gpt_label = (src_mask + [self.tokenizer.sep_token_id] + tgt \
            + [self.tokenizer.eos_token_id])[:self.max_len] #train: ignore src; inference: mask src
        gpt_attn = [1] * len(gpt_input)
        assert len(gpt_input) == len(gpt_label)
        if self.use_padding:
            input_padding = [self.tokenizer.pad_token_id] * (self.max_len - len(gpt_input))
            label_padding = [IGNORE_INDEX] * (self.max_len - len(gpt_input)) #self.tokenizer.pad_token_id
            attn_padding = [0] * (self.max_len - len(gpt_input))
            gpt_input.extend(input_padding), gpt_label.extend(label_padding), gpt_attn.extend(attn_padding)
        
        output = {
            'gpt_input': gpt_input,
            'gpt_label': gpt_label,
            'gpt_attn': gpt_attn,
        }
        # print(f'final:\nraw line[idx]: \n{self.lines[item]}\nbert_input: \n{bert_input}, \nbert_label: \n{bert_label}') # \nsegment_label: \n{segment_label}
        output = {k: torch.tensor(v) for k, v in output.items()}
        output['is_positive'] = self.is_pos
        
        return output
    
    def get_sent(self, idx, is_pos):
        if is_pos:
            t1, t2 = self.lines[idx]
        else:
            t1, t2 = self.lines[idx][0], self.lines[rand.randrange(len(self.lines))][1]
        return t1, t2
    
    def encode(self, sentence):
        sen_ids = self.tokenizer.encode(sentence)
        sen_mask = [self.tokenizer._vocab_str_to_int["[MASK]"]]*len(sen_ids)
        return sen_ids, sen_mask