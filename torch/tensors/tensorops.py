import torch 
import numpy as np


# move tensor to GPU if available 
#if torch.cuda.is_available():
#    tensor = tensor.to("cuda")
#else:
#    print('no cuda :D')
#
# numpy like indexing and sliciing 


tensor = torch.ones(4, 4)

# tensor[0] -> first row

# tensor[:,0] -> first column 

#tensor[:,-1] -> last column


# tensor[:,1] = 0
    # set  column to 0 


# join tensors 

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)



# Arithmetic operations 

# this computers matrix multiplcation between two tensors
# tensor.T returns transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)


y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
