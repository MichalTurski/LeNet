import torch.nn as nn
import torch.nn.functional as F
import copy

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convolution1 = nn.Conv2d(3, 8, 3, padding=1)
        self.convolution2 = nn.Conv2d(8, 15, 3, padding=1)
        self.convolution3 = nn.Conv2d(15, 25, 2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(25)
        self.convolution4 = nn.Conv2d(25, 25, 2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(25)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(25 * 64, 356)
        self.fc2 = nn.Linear(356, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, input):
        input = (F.relu(self.convolution1(input)))
        input = F.relu(self.convolution2(input))
        input = self.pool(self.batch_norm3(F.relu(self.convolution3(input))))
        input = self.pool(self.batch_norm4(F.relu(self.convolution4(input))))
        input = input.view(-1, 25 * 64)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        return F.relu(self.fc3(input))
