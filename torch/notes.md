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

# Datasets and DataLoaders 
## torch.utils.data.DataLoader
	- wraps an iterable around the Dataset to enable easy access to the samples
## torch.utils.data.Dataset
	- stores the samples and their corresponding labels
	- retrieves dataset's features and labels one sample at a time.


## Note 
	-pytorch domain libraries has a number o fpreloaded dataset that subclass inside ** torch.utils.data.Dataset ** and implement functions specific to particular data.
	- Used to prototype and benchmark model
		- https://pytorch.org/vision/stable/datasets.html


## Loading a Dataset
 -> see dataset.py

# Creating a Custom Dataset for your files 
A custom Dataset class must implement three functions:
	- __init__
		- initialize the directory containing the:
			- images 
			- annotations file 
			- both transforms* 
	
	- __len__
		- returns hte number of samples in our dataset
	- __getitem__
		- loads and returns a sample from the dataest at the given idx
			- identifies the image's location on disk 
			- converts that to a tensor using read_image
			- retrieves corresponding label from csv data
				- inside self.img_labels
			- calls the transform functions on them
				- if applicable 
			-> returns the tensor image and corresponding label in a tuple

## Preparing your data for training with DataLoaders
The `Dataset` retrieves our dataset's features and labels one sample at a time 

While trianingn a model, we want to pass minibatches of data and reshuffle the data at every epoch to reduce the model overfitting and use Python's `multiprocessing` to speed up data retrieval.  

For this  reason, `DataLoader` is an iterable that abstracts this complexity for us in an easy API.

## Iterate through DataLoader 
Each iteration returns a batch of `train_features` and `train_labels` (containgn 'batch_size = 64')


`shuffle = True`
	- after we iterate over all batches, data is shuffled 

## Dataset Styles
map-style datasets
	- one that implements __getitem__() and __len__() protocols and represnts a map from indices/keys to data samples 

iterable-style datasets
	- instance of subclass of IterableDataset that implements __iter__()
		- represents an iterable over data samples 
		- suitable for cases where random reads are expensive  / improbable (?)
			- or when batch size depends on fetched data.

### Samplers
For iterable-style datasheets, data loading order is entirely controlled by the user-defined iterable 
	- easier implementations of chunk reading and dynamic batch size 

map-style datasets 
	- torch.utils.data.Sampler
		- classes used to specify the sequence of indices/keys in data loading
		- represend iterable objects over indices to datasets 
		- `Sampler` cvould randomly permute a list of indices and yield each one at a time 
			- or yield a small number of htme for minibatch SGD
				- Stochastic gradient descent; split dataset into small subsets (batches) and compute gradients for each batch 


# Build the Neural Network
Neural networks are comprised on layers/modules that preofrm operations on data.
	- `torch.nn` namespace provides all the building blocks you need to build your own neural network 
		- every module in PyTorch subclasses `nn.Module`
			- A neural network is a module itself that consists of other modules/layers

Create a nn as a PythonClass with designated layers with 'nn.Sequential' {see neural.py}

## Calling the model 
Returns a 2-diemntional tensor with dim=0 corresponding to each output of 10 raw prediced values for each class, and dim = 1 corresponding to the indvidiual vlaues of each output 

We get predicition probabilities by passing it through an instance of nn.Softmax module


	- Pass tensors in model and nn.Softmax(dim = 1)(logits)
		- argmax(1)

## Model Layers 
`nn.Flatten`
	- intialize nn.Flatten layer to convert each 2D image into contiguous array of 784 pixel values (minibatch dimension when dim = 0 is maintained)

'nn.Linear'
	- applies linear transformation on the input using its stored weights and based
	- infeatures:int, out_features:int, bias=True, device=None, dtype=None
	- https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
'nn.ReLU'
	- non linear activatoins are what create complex mappings between the model's inputs and outputs. they are applied after linear transformations to introduce nonlinearity to help neural networks learn better (?)
	- other activations exists to introduce non-linearity in your model but using ReLU for example :D
'nn.Sequential'
	- ordered container of modules 
		- data is passed through all the modules in the same order as defined 
		- use sequential containers to put together quick network
			- like `seq_modules` (see neural.py)
'nn.Softmax'
	- last linear layer of neural networks returns 'logits'
		- raw values in [-infty, infty]
	- logits are scaled to values [0,1], representing model's predicted probabilites for each class 
	- 'dim' parameter indicates the dimension along which the values must sum to 1
		- normalizing results in batches (?)

## Model Parameters
Many layers inside a neural network are paramertized with associated weights / biases. 

Subclassing `nn.Module` automatically tracks all fields defined inside your model object
	- parameter's accessible using your model's `parameters()` or `named_parameters() methods`

# Remarks 
## Devices 
- CUDA 
	- Nvidia 
- MPS
	- MACOS  + METAL ( ? )
- CPU
