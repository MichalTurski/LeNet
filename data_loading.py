import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def prepare_data(batch_size, num_workers):
    null_transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         # transforms.RandomAffine(15, translate=(0.2, 0.2), scale=(0.9, 1.1)),
         transforms.RandomAffine(10, translate=(0.1, 0.1)),
         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
         transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                              (0.24703223, 0.24348513, 0.26158784))
         ])
        # [transforms.RandomHorizontalFlip(),
        #  transforms.RandomCrop(size=32, padding=[0, 2, 3, 4]),
        #  transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                              (0.24703223, 0.24348513, 0.26158784))])

    set_to_statisctics = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=null_transform)
    avg = (np.mean(set_to_statisctics.data, axis=(0, 1, 2))/255)
    std = (np.std(set_to_statisctics.data, axis=(0, 1, 2))/255)
    # print(avg, std)

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    return train_loader, test_loader, classes
