
'''
Generate the BinaryArith for some simple binary arithmetic tasks.
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
os.chdir('/home/user/shiqi/yichuan/EBM')
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

# Create directories and save tensors
splits = {
    "addition": generate_addition_samples(),
    "subtraction": generate_subtraction_samples(),
    "inversion": generate_inversion_samples()
}

for split_name, (x, y) in splits.items():
    split_path = f'./datasets/binary_arith/{split_name}'
    if not osp.exists(split_path):
        os.makedirs(split_path, exist_ok=True)
    torch.save(x, osp.join(split_path, "features.pt"))
    torch.save(y, osp.join(split_path, "labels.pt"))