import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


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


def plot(train_loss, test_loss, accuracy):
    plt.subplot(211)
    plt.plot(train_loss, linestyle='-.', label='training')
    plt.plot(test_loss, linestyle='-', label='test')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    axes = plt.gca()
    axes.set_ylim(bottom=0)
    plt.subplot(212)
    plt.plot(accuracy, linestyle='-', label='training')
    axes = plt.gca()
    axes.set_ylim(bottom=0)
    plt.show()


def roc_curves(net, test_loader, classes, device):
    with torch.no_grad():
        outputs = []
        labels_vec = []
        for (inputs, labels) in test_loader:
            inputs = inputs.to(device)
            outputs_cpu = net(inputs).cpu()
            for output in outputs_cpu:
                outputs.append(output.detach().numpy())
            for label in labels:
                labels_vec.append(label.detach().numpy())
    outputs_array = np.array(outputs)
    labels_array = np.array(labels_vec)
    for i in range(10):
        digit_pred = outputs_array[:, i]
        y_expected = labels_array == i
        fpr, tpr, thresholds = roc_curve(y_expected, digit_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.9, color='r', label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='g', label='Random classifier', alpha=0.4)
        plt.title(f"ROC curve for class {classes[i]}, AUC = {roc_auc}")
        plt.legend(loc='lower right')
        plt.show()

