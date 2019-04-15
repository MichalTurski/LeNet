import data_loading
import network
import checks
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pickle

##########
# Params #
##########
epochs = 90
batch_size = 64
num_workers = 8
learning_rate1 = 0.001
learning_rate2_epoch = 30
learning_rate2 = 0.0001
learning_rate3_epoch = 60
learning_rate3 = 0.00001
momentum = 0.9
verbose = True
check_period = 750
loss_rise_threshold = 20
# net = network.LeNet()
# net = pickle.load(open("best_net_unfreezed1.pickle", "rb"))
net = network.ResNet()
# net = network.ResNet34()
# net = network.ResNet18_extended()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = data_loading.prepare_data(batch_size, num_workers)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate1, momentum=momentum)
loss_function = nn.CrossEntropyLoss()

train_loss = checks.test_loss(train_loader, net, device, loss_function)
train_loss_list = [train_loss]
test_loss = checks.test_loss(test_loader, net, device, loss_function)
test_loss_list = [test_loss]
accuracy = checks.accuracy(test_loader, net, device)
accuracy_list = [accuracy]
if verbose:
    print(f'[epoch {0}] train loss = {train_loss:.3f}, '
          f'test loss = {test_loss:.3f}, accuracy = {accuracy * 100:.2f}%')

prev_loss = float("inf")
lowest_loss = float("inf")
loss_rise_count = 0
best_net = copy.deepcopy(net)

for epoch in range(epochs):
    if epoch == learning_rate2_epoch:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate2, momentum=momentum)
    if epoch == learning_rate3_epoch:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate3, momentum=momentum)
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
    train_loss = train_loss/i
    train_loss_list.append(train_loss)
    test_loss = checks.test_loss(test_loader, net, device, loss_function)
    test_loss_list.append(test_loss)
    accuracy = checks.accuracy(test_loader, net, device)
    accuracy_list.append(accuracy)
    if verbose:
        print(f'[epoch {epoch+1}] train loss = {train_loss:.3f}, '
              f'test loss = {test_loss:.3f}, accuracy = {accuracy*100:.2f}%')
    if test_loss > prev_loss:
        loss_rise_count += 1
        if loss_rise_count >= loss_rise_threshold:
            break
    else:
        loss_rise_count = 0

    if test_loss < lowest_loss:
        best_net = copy.deepcopy(net)
        lowest_loss = test_loss

    prev_loss = test_loss
checks.plot(train_loss_list, test_loss_list, accuracy_list)
checks.roc_curves(best_net, test_loader, classes, device)
pickle.dump(best_net, open("best_net.pickle", "wb"))
pickle.dump(train_loss_list, open("train_loss.pickle", "wb"))
pickle.dump(test_loss_list, open("test_loss.pickle", "wb"))
pickle.dump(accuracy_list, open("accuracy.pickle", "wb"))

