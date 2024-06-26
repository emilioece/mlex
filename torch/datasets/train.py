from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# load data 
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
# easier api to use 
train_dataloader = DataLoader(training_data, batch_size=64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)


# display image and label 
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap = "gray")
plt.show()
print(f'label: {label}')
