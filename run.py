import data_loading
import network
import checks
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

##########
# Params #
##########
epochs = 40
batch_size = 32
num_workers = 6
learning_rate = 0.001
momentum = 0.9
verbose = True
check_period = 750

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = data_loading.prepare_data(batch_size, num_workers)
net = network.LeNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

train_loss_list = [checks.test_loss(train_loader, net, device, loss_function)]
test_loss_list = [checks.test_loss(test_loader, net, device, loss_function)]
accuracy_list = [checks.accuracy(test_loader, net, device)]

for epoch in range(epochs):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % check_period == (check_period - 1):
            train_loss = train_loss/check_period
            train_loss_list.append(train_loss)
            test_loss = checks.test_loss(test_loader, net, device, loss_function)
            test_loss_list.append(test_loss)
            accuracy_list.append(checks.accuracy(test_loader, net, device))
            if verbose:
                print(f'[epoch {epoch+1}, batch {i+1}] train loss = {train_loss:.3f}, '
                      f'test loss = {test_loss:.3f}')
            train_loss = 0.0

plt.subplot(211)
plt.plot(train_loss_list, linestyle='-.', label='training')
plt.plot(test_loss_list, linestyle='-', label='test')
plt.legend()
plt.ylabel('loss')
plt.xlabel('epochs')
axes = plt.gca()
axes.set_ylim(bottom=0)
plt.subplot(212)
plt.plot(accuracy_list, linestyle='-', label='training')
axes = plt.gca()
axes.set_ylim(bottom=0)
plt.show()
