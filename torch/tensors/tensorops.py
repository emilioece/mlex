import torch 
import numpy as np


# move tensor to GPU if available 

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f'device tensor is stored on: {tensor.device}')
