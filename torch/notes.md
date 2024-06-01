(Working in tensors/tensors.py)
# Tensors 
Specialized data structure that are similiar to arrays and matrices 
	- used to encode inputs and output 
		- as well as model's parameters

Similar to NumPy's ndarrays 
		- Tensors can run on GPUS to accelerate computing


Tensors can be intialized
	- directly from data 
	- from numpy arrays 
	- from another tensor
		- retains the properties of the argument tensors 
		like shape / datatype
	- with random or constant values 
		- shape() is a tyuple of tensor dimensions

## Tensor Attributes 

Describe their shape, datatype, and the device on whcih they are stored.


(Working in tensors/tensorops.py)
# Tensor Operations
Over 100 tensor operations described in https://pytorch.org/docs/stable/torch.html


Each of them can be run on the GPU, typically higher speeds that on a CPU 
	- ON COLAB, allocate a GPu by going to 
		- Edit -> Notebook settings 

