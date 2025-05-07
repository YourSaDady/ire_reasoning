
'''
Generate some simple decimal arithmetic tasks.
format:
    - x: 12-bits
        2-bits split-indecator: {'00': Addition, '01': Substraction, '10': Inversion}
        10-bits values: either x1 (5-bits) and x2 (5-bits) for addition and substraction, or x (10-bits) for inversion
    - y: 10-bits

split size: 2000 samples
'''

import random
import torch
import os
import os.path as osp
from tqdm import tqdm
os.chdir('/home/yichuan/HKU/EBM')
print(f'The current working directory: {os.getcwd()}')

def generate_addition_samples_old(num_samples=2000):
    features, labels = [], []
    for _ in range(num_samples):
        x1 = random.randint(0, 31)
        x2 = random.randint(0, 31)
        sum_num = x1 + x2
        # Convert to 12-bit x (00 + x2 + x1)
        x_bin = '00' + format(x2, '05b') + format(x1, '05b')
        # Convert to 10-bit y (sum)
        y_bin = format(sum_num, '010b')
        # Convert binary strings to integer tensors
        features.append([int(bit) for bit in x_bin])
        labels.append([int(bit) for bit in y_bin])
    return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)

def generate_subtraction_samples_old(num_samples=2000):
    features, labels = [], []
    for _ in range(num_samples):
        x1 = random.randint(0, 31)
        x2 = random.randint(0, x1)  # Ensure x2 <= x1
        y_num = x1 - x2
        x_bin = '01' + format(x2, '05b') + format(x1, '05b')
        y_bin = format(y_num, '010b')
        features.append([int(bit) for bit in x_bin])
        labels.append([int(bit) for bit in y_bin])
    return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)

def generate_inversion_samples_old(num_samples=2000):
    features, labels = [], []
    for _ in range(num_samples):
        x_bits = [random.choice([0, 1]) for _ in range(10)]
        y_bits = [1 - bit for bit in x_bits]  # Flip bits
        # Prepend split indicator '10' to x
        x_full = [1, 0] + x_bits  # 2 bits (10) + 10 bits = 12 bits
        features.append(x_full)
        labels.append(y_bits)
    return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)


def generate_addition_samples():
    features, labels = [], []
    for x1 in range(32):  # 0 to 31
        for x2 in range(32):  # 0 to 31
            sum_num = x1 + x2
            x_bin = '00' + format(x2, '05b') + format(x1, '05b')
            y_bin = format(sum_num, '010b')
            features.append([int(bit) for bit in x_bin])
            labels.append([int(bit) for bit in y_bin])
    return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)

def generate_subtraction_samples():
    features, labels = [], []
    for x1 in range(32):
        for x2 in range(x1 + 1):  # Ensure x2 <= x1
            y_num = x1 - x2
            x_bin = '01' + format(x2, '05b') + format(x1, '05b')
            y_bin = format(y_num, '010b')
            features.append([int(bit) for bit in x_bin])
            labels.append([int(bit) for bit in y_bin])
    return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)

def generate_inversion_samples():
    features, labels = [], []
    for num in range(1024):  # 2^10 = 1024 possible 10-bit values
        x_bits = [(num >> i) & 1 for i in range(9, -1, -1)]  # Convert to 10-bit list
        y_bits = [1 - bit for bit in x_bits]
        x_full = [1, 0] + x_bits  # Prepend '10' split indicator
        features.append(x_full)
        labels.append(y_bits)
    return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)

'''
Larger Binary-arithmetics, overall sequence length is 26-bits.
Save in txt format, separate string x and y with " +++$+++ "
Each sample pair: ('0 1 0 ...' + " +++$+++ " + '1 0 0 ...')

Addition: x1 + x2 = y
    - flag: '00'
    - x1: 8-bits
    - x2: 8-bits
    - y: 9-bits

Subtraction: x1 - x2 = y
    - flag: '01'
    - x1: 9-bits
    - x2: 8-bits
    - y: 8-bits

Inversion: !x = y
    - flag: '10'
    - x: 12-bits
    - y: 12-bits
'''
def generate_addition_samples_txt(save_in_txt=True): 
    features, labels = [], []
    for x1 in tqdm(range(2**8), desc='addition'): # 0 to 2^8
        for x2 in range(2**8): # 0 to 2^8
            y = x1 + x2
            x_bin = '00' + format(x1, '08b') + format(x2, '08b')
            y_bin = format(y, '09b')
            if save_in_txt:
                features.append(' '.join(x_bin))
                labels.append(' '.join(y_bin))
            else:
                features.append([int(bit) for bit in x_bin])
                labels.append([int(bit) for bit in y_bin])
    if save_in_txt:
        return features, labels
    else:
        return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)
    
def generate_subtraction_samples_txt(save_in_txt=True): 
    features, labels = [], []
    for x2 in tqdm(range(2**8), desc='subtraction'): # 0 to 2^8
        for y in range(2**8): # 0 to 2^8
            x1 = x2 + y #y = x1 - x2
            x_bin = '01' + format(x1, '09b') + format(x2, '08b')
            y_bin = format(y, '08b')
            if save_in_txt:
                features.append(' '.join(x_bin))
                labels.append(' '.join(y_bin))
            else:
                features.append([int(bit) for bit in x_bin])
                labels.append([int(bit) for bit in y_bin])
    if save_in_txt:
        return features, labels
    else:
        return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)
    
def generate_inversion_samples_txt(save_in_txt=True):
    features, labels = [], []
    for num in tqdm(range(2**12), desc='inversion'):  # 2^12 possible values
        x_bits_str = bin(num)[2:].zfill(12)
        x_bits = ' '.join('10' + x_bits_str) 
        y_bits = ' '.join('1' if bit == '0' else '0' for bit in x_bits_str)
        if save_in_txt:
            features.append(x_bits)
            labels.append(y_bits)
        else:
            features.append([int(bit) for bit in x_bits.split()])
            labels.append([int(bit) for bit in y_bits.split()])
    if save_in_txt:
        return features, labels
    else:
        return torch.tensor(features, dtype=torch.int8), torch.tensor(labels, dtype=torch.int8)



save_in_txt = True
train_val_rate = 0.9

# Create directories and save tensors
splits = {
    "addition": generate_addition_samples_txt(save_in_txt),
    "subtraction": generate_subtraction_samples_txt(save_in_txt),
    "inversion": generate_inversion_samples_txt(save_in_txt)
}

for split_name, (xs, ys) in splits.items():
    x_ys = list(zip(xs, ys))
    random.shuffle(x_ys)
    split_path = f'./datasets/binary_arith_txt/{split_name}'
    if not osp.exists(split_path):
        os.makedirs(split_path, exist_ok=True)
    if save_in_txt:
        train_num = int(len(xs)*0.9)
        train_data = x_ys[:train_num]
        val_data = x_ys[train_num:]
        with open(osp.join(split_path, 'train.txt'), 'w') as train_file:
            for x_y in train_data:
                train_file.write(' +++$+++ '.join(x_y) + '\n') #replaced by [UNK]
        with open(osp.join(split_path, 'val.txt'), 'w') as val_file:
            for x_y in val_data:
                val_file.write(' +++$+++ '.join(x_y) + '\n')
        print(f'\n{split_name} datasets \ntrain({len(train_data)}) and val ({len(val_data)}) saved to {split_path}')
    else:
        torch.save(x_ys[0], osp.join(split_path, "features.pt"))
        torch.save(x_ys[1], osp.join(split_path, "labels.pt"))