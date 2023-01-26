import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


train_dataset = torchvision.datasets.FashionMNIST("./data", download=False,
                                                  transform=transforms.Compose([transforms.ToTensor()])) # ToTensor : convert images to tensor
test_dataset = torchvision.datasets.FashionMNIST("./data", download=False, train=False,
                                                 transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=100)
test_loader = DataLoader(test_dataset, batch_size=100)


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(dim=0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


device = torch.device('cpu')
epochs = 10
losses = []
accuracy = []
model = FashionMNISTCNN()
optimizer = Adam(model.parameters())

for i, epoch in enumerate(range(epochs)):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # clear gradients
        optimizer.zero_grad()
        # define model and prediction
        outputs = model(x)
        # get loss
        loss = F.cross_entropy(outputs, y)
        # back-propagation
        loss.backward()
        # update parameters based on gradients
        optimizer.step()

    # Test after per epoch
    total, correct = 0, 0
    for x, y in test_loader:
        outputs = model(x)
        test_loss = F.cross_entropy(outputs, y)
        predictions = torch.max(outputs, 1)[1].to(device)   # torch.max returns tuple.
        correct += (predictions == y).sum()
        total += len(y)

    accuracy = correct * 100 / total
    print(f"Iteration:{i+1}, Train Loss:{loss}, Test Loss:{test_loss}, Accuracy for Test:{accuracy}")


