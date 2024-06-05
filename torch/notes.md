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
