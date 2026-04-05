import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

torchvision.datasets.FashionMNIST(root="/data", train=True, download=True, transform=transforms.ToTensor())



transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Transformation pipeline defined successfully.")

# Load the training dataset with the defined transformation
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
print("FashionMNIST training dataset loaded successfully.")

# Define transformation for the test set
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the testing dataset with the test transformation
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)
print("FashionMNIST testing dataset loaded successfully.")

# Define batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoaders created successfully.")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# Get the shape of a sample image
# The transform applied to train_dataset makes it a tensor, so we can access its shape directly
sample_image, sample_label = train_dataset[0]
print(f"Shape of a sample image: {sample_image.shape}")

# Get the first batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Map integer labels to class names for better readability
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Display a selection of images and their labels
fig = plt.figure(figsize=(10, 8))
for idx in np.arange(6):
    ax = fig.add_subplot(2, 3, idx + 1, xticks=[], yticks=[])
    # Unnormalize the image for display
    img = images[idx].numpy()
    # Reverse normalization: img = img * std + mean
    mean = 0.1307
    std = 0.3081
    img = img * std + mean
    img = np.clip(img, 0, 1) # Clip values to [0,1] after unnormalization
    ax.imshow(np.squeeze(img), cmap='gray')
    ax.set_title(class_names[labels[idx]])
plt.tight_layout()
plt.show()

print("Displayed a selection of images and their labels from the first mini-batch.")