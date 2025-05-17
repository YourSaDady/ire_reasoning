import torch
print("PyTorch version:", torch.__version__) #2.6.0+cpu
print("CUDA available:", torch.cuda.is_available()) #False
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)