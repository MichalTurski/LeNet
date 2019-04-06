import torch.nn as nn
import torch.nn.functional as F
import copy

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convolution1 = nn.Conv2d(3, 8, 5)
        self.convolution2 = nn.Conv2d(8, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 25, 140)
        self.fc2 = nn.Linear(140, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        input = self.pool(F.relu(self.convolution1(input)))
        input = self.pool(F.relu(self.convolution2(input)))
        input = input.view(-1, 20 * 25)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        return F.relu(self.fc3(input))
