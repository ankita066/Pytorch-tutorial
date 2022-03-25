#Imports
from random import shuffle
from turtle import forward

from zmq import device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size, classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

device = torch.device('cpu')

#Hyper parameters
input_size = 784
classes = 10
learning_rate = 0.01
epochs = 1
batch_size = 64

train_data = datasets.MNIST(train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True)


