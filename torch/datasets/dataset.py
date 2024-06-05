import torch 
from torch.utils.data import Dataset
from torchvision import datasets 
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# downloading training and test data 
training_data = datasets.FashionMNIST(
        root = "data",
        # :3
        train = True,
        download = True,
        transform=ToTensor()
        )

test_data = datasets.FashionMNIST(
        root="data",
        train = False, 
        download=True, 
        transform=ToTensor()
        )

# length of data
train_len = len(training_data)
test_len = len(test_data)

# index datasets like a list and use matplot lib to visualze samples
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# figure to display items
figure = plt.figure(figsize = (8,8))
cols, rows = 3, 3

# iterate through dataset
for i in range(1, cols*rows + 1):
    sample_index = torch.randint(train_len, size=(1,)).item()
    img, label = training_data[sample_index]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
