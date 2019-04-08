import torch.nn as nn
import torch.nn.functional as F
import copy

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convolution1 = nn.Conv2d(3, 8, 3, padding=1)
        self.convolution2 = nn.Conv2d(8, 20, 3, padding=1)
        self.convolution3 = nn.Conv2d(20, 30, 2, padding=1)
        self.convolution4 = nn.Conv2d(30, 40, 2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40 * 16, 196)
        self.fc2 = nn.Linear(196, 86)
        self.fc3 = nn.Linear(86, 10)

    def forward(self, input):
        input = F.relu(self.convolution1(input))
        input = F.relu(self.convolution2(input))
        input = self.pool(input)
        input = self.pool(F.relu(self.convolution3(input)))
        input = (self.convolution4(input))
        input = F.relu(input)
        input = self.pool(input)
        input = input.view(-1, 40 * 16)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        return F.relu(self.fc3(input))
