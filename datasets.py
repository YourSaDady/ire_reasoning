import os.path as osp
import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
sys.path.append('/home/user/shiqi/yichuan/EBM/ire_reasoning')
os.chdir('/home/user/shiqi/yichuan/EBM/ire_reasoning')
# print(f'The current working directory: {os.getcwd()}')

'''
Sudoku Task. Borrowed from IRED
'''
def get_data_dir(identifier):
    base_dir = osp.join(osp.dirname(__file__), 'datasets') # ./datasets??
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

def _rescale(x):
    return (x - 0.5) * 2


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
        else:
            self.features = self.features[int(nr_datapoints * 0.9):]
            self.labels = self.labels[int(nr_datapoints * 0.9):]

        if dataset_identifier == 'sudoku':
            self.cond_entry = (self.features.sum(axis=-1) == 1)[:, :, :, None].expand(-1, -1, -1, 9)
        self.inp_dim = self.features[0].numel()
        self.out_dim = self.labels[0].numel()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.task == 'sudoku':
            return _rescale(self.features[idx].reshape(-1)), _rescale(self.labels[idx].reshape(-1)), self.cond_entry[idx].reshape(-1)
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


def batchlize(dataset, batch_size=1000):
    dataset = [torch.cat([data[0].unsqueeze(0), data[1].unsqueeze(0)], dim=1) for data in dataset] #TODO: binary-arith only (for now)
    return [torch.cat(dataset[idx:idx+batch_size], dim=0) for idx in range(0, len(dataset), batch_size)]



def load_data(task, train_batch_size, val_batch_size):
    if task == 'sudoku' or task.startswith('binary'):
        train_set = SATDataset(task, split='train')
        val_set = SATDataset(task, split='val')
        print(f'train_set.inp_dim: {train_set.inp_dim}') #12
        print(f'train_set.out_dim: {train_set.out_dim}') #10
        print(f'first sample: {train_set[0]}') #(tensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=torch.int8), tensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=torch.int8))
        print(f'train_size: {len(train_set)}, val_size: {len(val_set)}') #1800, 200
        return batchlize(train_set, train_batch_size), batchlize(val_set, val_batch_size), len(train_set), len(val_set)
    else:
        raise NotImplementedError(f'The specified task: {task} is not defined')