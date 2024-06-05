import os 
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Define neural network by subclassing nn.Module and iniatlize neural netowrk layers in init 

# every 'nn.Module' subclass implements the oepartions on input data in the forward method 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
                )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# sample mini nbatch of 3 images of size 28x28
input_image = torch.rand(3, 28, 28)
print(input_image.size())


# Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f'flatten layer size: {flat_image.size()}')


# Linear layer 
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f'hidden layer size: {hidden1}')

# Non linear activation
print(f'{hidden1}\n\n')
hidden1 = nn.ReLU()(hidden1)
print(f'{hidden1}\n\n')





if True:
    model = NeuralNetwork().to(device)
    # to use the model, pass input data and it will execute the model's
    # forward along with some background operations 
            # note: dont call model.forward() directly
    #print(model)

    X = torch.rand(1, 28, 28, device = device)
    logits = model(X)
    
    pred_probab = nn.Softmax(dim=1)(logits)
    print(f'pred prob : {pred_probab}')

    y_pred = pred_probab.argmax(1)
    print(f'predicted class : {y_pred}')

seq_modules = nn.Sequential(
        flatten, 
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10),
        )
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
#print(pred_probab)


# Model parameters
print(f'\n\nmodel structure: {model}\n\n')

for name, param in model.named_parameters():
    print(f'layer: {name} | size: {param.size()} | values: {param[:2]}\n')


