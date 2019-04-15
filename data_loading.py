import torch
import torchvision
import torchvision.transforms as transforms


def prepare_data(batch_size, num_workers):
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         # transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
         transforms.RandomAffine(10, translate=(0.1, 0.1)),
         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
        # [transforms.RandomHorizontalFlip(),
        #  transforms.RandomCrop(size=32, padding=[0, 2, 3, 4]),
        #  transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
