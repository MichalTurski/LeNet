import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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

        total_params = sum(p.numel() for p in self.parameters())
        print(f'{total_params:,} total parameters.')

    def forward(self, input):
        input = (F.relu(self.convolution1(input)))
        input = F.relu(self.convolution2(input))
        input = self.pool(self.batch_norm3(F.relu(self.convolution3(input))))
        input = self.pool(self.batch_norm4(F.relu(self.convolution4(input))))
        input = input.view(-1, 25 * 64)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        return F.relu(self.fc3(input))


def ResNet():
    net = models.resnet18(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    #
    # for param in net.layer4.parameters():
    #     param.requires_grad = True

    net.fc = nn.Linear(512, 10)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    return net


# def VGG():
#     net = models.vgg11(pretrained=True)
#     for layer in net.features():
#         for param in layer:
#             param.requires_grad = False
#
#     net.classifier[6] = nn.Linear(4096, 10)
#
#     total_params = sum(p.numel() for p in net.parameters())
#     print(f'{total_params:,} total parameters.')
#     total_trainable_params = sum(
#         p.numel() for p in net.parameters() if p.requires_grad)
#     print(f'{total_trainable_params:,} training parameters.')
#
#     return net
