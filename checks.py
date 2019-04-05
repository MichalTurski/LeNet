import torch


def test_loss(test_loader, net, device, loss_function):
    with torch.no_grad():
        loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss += loss_function(outputs, labels)
        return loss/i


def accuracy(test_loader, net, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for (inputs, labels) in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct/total
