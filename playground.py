import torch
# print("PyTorch version:", torch.__version__) #2.6.0+cpu
# print("CUDA available:", torch.cuda.is_available()) #False
# if torch.cuda.is_available():
#     print("CUDA version:", torch.version.cuda)
t1 = torch.tensor([ 7,  8, 20,  7, 10, 20, 11, 20,  9,  7,  1,  2,  2,  2,  2,  2,  2,  2, \
         2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0, \
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
t2 = torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 27, 28, 29, 30, 31, 32, 33, \
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
result = t1[t2]

print(f'result.shape: {result.shape}')
print(f'result: {result}')