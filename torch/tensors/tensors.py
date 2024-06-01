import torch 
import numpy as np

# Tensors can be intialized directly from data 

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)


# from NumPy arrays 
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)


# from another tensor 
    # new tensor retains the properties of the argument tensors 
             # shape  / datatype 

x_ones = torch.ones_like(x_data)
#print(f'ones tensor \n {x_ones} \n')


x_rand = torch.rand_like(x_data, dtype=torch.float)

#print(f'random tensor \n {x_rand} \n')


# with random or c onstant values 
# shape is a tuple of tensor dimensions 

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


#print(f'random tensor: \n {rand_tensor} \n')
#print(f'ones tensor: \n {ones_tensor} \n')
#print(f'zeros tensor: \n {zeros_tensor} \n')


# Tensor Attributes 

tensor = torch.rand(3, 4)
print(f'shape of tensor: {tensor.shape}')
print(f'datatype of tensor: {tensor.dtype}')
print(f'device tensor is stored on: {tensor.device}')

