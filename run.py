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
epochs = 15
batch_size = 32
num_workers = 4
learning_rate = 0.005
momentum = 0.9
verbose = True
check_period = 750
loss_rise_threshold = 5
# net = network.LeNet()
net = pickle.load(open( "best_net.pickle", "rb"))
# net = network.ResNet()
# net = network.VGG()
# net = models.resnet18(pretrained=True)
# net.fc = nn.Linear(512, 10)
# net = models.AlexNet()
# net.classifier[6] = nn.Linear(4096, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = data_loading.prepare_data(batch_size, num_workers)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
# checks.roc_curves(best_net, test_loader, classes, device)
pickle.dump(best_net, open("best_net.pickle", "wb"))

