import os.path as osp
import sys
import os
import torch
import pandas as pd
import numpy as np
sys.path.append('/home/yichuan/HKU/EBM/ire_reasoning')
os.chdir('/home/yichuan/HKU/EBM/ire_reasoning')
# print(f'The current working directory: {os.getcwd()}')

from models.bert import train_tokenizer, BERTDataset
from datasets import Dataset, Dataloader

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

def load_satnet_dataset(data_dir):
    if not osp.exists(data_dir):
        raise ValueError(f'Data directory {data_dir} does not exist. Run data/download-satnet.sh to download the dataset.')
    features = torch.load(osp.join(data_dir, 'features.pt'))
    labels = torch.load(osp.join(data_dir, 'labels.pt'))
    return features, labels

class SudokuDataset(Dataset):
    def __init__(self, dataset_identifier, split): #identifier就是'sudoku', split = {train, val}
        self.features, self.labels = load_satnet_dataset(get_data_dir(dataset_identifier))
        nr_datapoints = len(self.features)

        assert split in ('train', 'val')
        self.split = split
        if self.split == 'train':
            self.features = self.features[:int(nr_datapoints * 0.9)]
            self.labels = self.labels[:int(nr_datapoints * 0.9)]
        else:
            self.features = self.features[int(nr_datapoints * 0.9):]
            self.labels = self.labels[int(nr_datapoints * 0.9):]

        self.cond_entry = (self.features.sum(axis=-1) == 1)[:, :, :, None].expand(-1, -1, -1, 9) #?
        self.inp_dim = self.features[0].numel()
        self.out_dim = self.labels[0].numel()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return _rescale(self.features[idx].reshape(-1)), _rescale(self.labels[idx].reshape(-1)), self.cond_entry[idx].reshape(-1)

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




# '''
# Simplyfied version from MDLM
# '''
# def get_dataloaders(config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None):
#     #TODO: simplify to Sudoku and Binary tasks
#     return train_loader, val_loader

# def get_dataset(
#     dataset_name, tokenizer, wrap, mode, cache_dir,
#     block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False):
#     if wrap:
#         filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
#     else:
#         filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
#     _path = os.path.join(cache_dir, filename)
    
#     if utils.fsspec_exists(_path):
#         LOGGER.info(f'Loading data from: {_path}')
#         return datasets.load_from_disk(_path).with_format('torch')
#     LOGGER.info(f'Generating new data at: {_path}')
#     #...


# def get_tokenizer(config):
#   if config.data.tokenizer_name_or_path == 'text8':
#     tokenizer = Text8Tokenizer()
#   elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
#     tokenizer = transformers.BertTokenizer.\
#       from_pretrained('bert-base-uncased')
#   else:
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#       config.data.tokenizer_name_or_path)

#   if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
#       or isinstance(tokenizer, transformers.GPT2Tokenizer)):
#     tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
#       (tokenizer.bos_token, tokenizer.bos_token_id),
#       (tokenizer.eos_token, tokenizer.eos_token_id))

#   # For wrapped batches:
#   #  [BOS] sent1 [EOS] sent2-fragment [EOS]
#   #  [BOS] sent2-fragment [EOS] sent3 [EOS]
#   if tokenizer.bos_token is None:
#     if tokenizer.cls_token is None:
#       raise AttributeError(
#         'Tokenizer must have a bos_token or '
#         f'cls_token: {tokenizer}')
#     tokenizer.bos_token = tokenizer.cls_token
#   if tokenizer.eos_token is None:
#     if tokenizer.sep_token is None:
#       raise AttributeError(
#         'Tokenizer must have a eos_token '
#         f'or sep_token: {tokenizer}')
#     tokenizer.eos_token = tokenizer.sep_token
#   if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#   return tokenizer

# #borrowed
# def get_dataloaders_org(config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None):
#     num_gpus = torch.cuda.device_count()
#     assert (config.loader.global_batch_size
#           == (config.loader.batch_size
#               * config.trainer.num_nodes
#               * num_gpus
#               * config.trainer.accumulate_grad_batches))
#     if config.loader.global_batch_size % (
#         num_gpus * config.trainer.accumulate_grad_batches) != 0:
#         raise ValueError(
#             f'Train Batch Size {config.training.batch_size}'
#             f'not divisible by {num_gpus} gpus with accumulation '
#             f'{config.trainer.accumulate_grad_batches}.')
#     if config.loader.eval_global_batch_size % num_gpus != 0:
#         raise ValueError(
#             f'Eval Batch Size for {config.eval.batch_size} '
#             f'not divisible by {num_gpus}.')
#     if skip_train:
#         train_set = None
#     else:
#         train_set = get_dataset(
#             config.data.train,
#             tokenizer,
#             mode='train',
#             wrap=config.data.wrap,
#             cache_dir=config.data.cache_dir,
#             block_size=config.model.length)
    
#     if config.data.valid in ['text8', 'lm1b', 'ag_news']:
#         validation_split = 'test'
#     else:
#         validation_split = 'validation'
#     if skip_valid:
#         valid_set = None
#     else:
#         valid_set = get_dataset(
#             config.data.valid,
#             tokenizer,
#             wrap=config.data.wrap,
#             mode=validation_split,
#             cache_dir=config.data.cache_dir,
#             block_size=config.model.length,
#         streaming=False)

#     if skip_train:
#         train_loader = None
#     else:
#         train_loader = torch.utils.data.DataLoader(
#         train_set,
#         batch_size=config.loader.batch_size,
#         num_workers=config.loader.num_workers,
#         pin_memory=config.loader.pin_memory,
#         shuffle=not config.data.streaming,
#         persistent_workers=True)
#         train_loader.tokenizer = tokenizer
#     if skip_valid:
#         valid_loader = None
#     else:
#         if valid_seed is None:
#             shuffle_valid = False
#             generator = None
#         else:
#             shuffle_valid = True
#             generator = torch.Generator().manual_seed(valid_seed)
#             valid_loader = torch.utils.data.DataLoader(
#             valid_set,
#             batch_size=config.loader.eval_batch_size,
#             num_workers=config.loader.num_workers,
#             pin_memory=config.loader.pin_memory,
#             shuffle=shuffle_valid,
#             generator=generator)
#             # Will be used in generative perplexity calculation
#             valid_loader.tokenizer = tokenizer

#     return train_loader, valid_loader

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



def load_data(task, train_batch_size, val_batch_size):
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
    
def load_bert_data(task, stage, max_len, train_batch_size, val_batch_size):
    '''
    params:
        max_len: the padding length to the BERTDataset, the same parameter for initializing BERT
    '''
    print(f'Now start training tokenizer...')
    tokenizer = train_tokenizer(task)
    # tokenizer = Tokenizer.from_file('./ire_reasoning/models/tokenizer_{task}.json')
    
    if task.startswith('binary'):
        if task.endswith('addition'):
            train_path, test_path = '../datasets/binary_arith_txt/addition/train.txt', '../datasets/binary_arith_txt/addition/test.txt'
            train_pairs, test_pairs = [], []
            for path, pairs in zip([train_path, test_path], [train_pairs, test_pairs]):
                if path.endswith('train.txt'):
                    tag = 'train'
                else:
                    tag = 'test'
                with open(path, 'r') as file:
                    for line in file:
                        t1, t2 = line.strip().split(' +++$+++ ')
                        pairs.append((t1, t2))
                
                print(f'\n{tag} set size: {len(pairs)}')
            train_data, test_data = BERTDataset(train_pairs, stage=stage, max_len=max_len, tokenizer=tokenizer), \
                BERTDataset(test_pairs, stage=stage, max_len=max_len, tokenizer=tokenizer)
            train_loader, test_loader = Dataloader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True), \
                Dataloader(test_data, batch_size=val_batch_size, shuffle=True, pin_memory=True)
        return train_loader, test_loader, len(train_data), len(test_data)

    else:
        raise NotImplementedError