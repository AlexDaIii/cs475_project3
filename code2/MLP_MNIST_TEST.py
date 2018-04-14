import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math

# TODO: If have time try adding stuff to output to tensorboard
try:
    from tensorboardX import SummaryWriter

    tb = True
except ModuleNotFoundError:
    tb = False

# TODO: Import data here:

train_batch_size = 32
test_batch_size = 1000
# creates a data loader - downloads it into ./data and transforms it to pytorch tensor and normalizes it to gaussian
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                                         transform=transforms.Compose([transforms.ToTensor()])),
                          batch_size=test_batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor()])),
                         batch_size=train_batch_size, shuffle=False)


##

# TODO: Define your model:
class Net(nn.Module):
    def __init__(self, n_nodes, n_classes):
        super(Net, self).__init__()
        # define the architecture
        self.fc1 = nn.Linear(28 * 28, n_nodes[0])  # hl1
        self.fc2 = nn.Linear(n_nodes[0], n_nodes[1])  # hl2
        self.fc3 = nn.Linear(n_nodes[1], n_nodes[2])  # hl3
        self.output = nn.Linear(n_nodes[2], n_classes)

        # Define layers making use of torch.nn functions:

    def forward(self, x):
        # flatten image
        net = x.view(-1, 28 * 28)

        # Define how forward pass / inference is done:
        net = self.fc1(net)
        net = F.relu(net)
        net = self.fc2(net)
        net = F.relu(net)
        net = self.fc3(net)
        net = F.relu(net)
        net = self.output(net)

        # return output
        return net


# 10 classes for 10 digits
n_classes = 10
# some hyperparameters
n_nodes_hl = [500, 500, 500]
# hyperparameters for Adam optimizer - these are just the default of the Adam
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08

# get the network
# print("Setting Net")
model = Net(n_nodes=n_nodes_hl, n_classes=n_classes)
# define loss and optimizer
# print("Setting optimizers")
criteria = nn.CrossEntropyLoss()  # This does softmax and cross entropy w logits
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)


# TODO: Train your model:
def train(num_epoch=10, show_metric=False):
    #print("Training")
    model.train()  # set model to train mode

    num_iter = math.ceil(len(train_loader.dataset) / test_batch_size)

    # get the batch index and a tuple of x, y
    for epoch in range(0, num_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Convert torch tensor to Variable
            data, target = Variable(data), Variable(target)
            # set grad to 0 bc it accumulates gradients
            optimizer.zero_grad()
            # does model.forward - DO NOT USE model.forward
            output = model(data)
            # compute loss according to the loss function
            loss = criteria(output, target)
            # backprop based on loss
            loss.backward()
            # do next step
            optimizer.step()

            if show_metric:
                print("Epoch: " + str(epoch) + " Batch: " + str(batch_idx) +
                      "/" + str(num_iter) + ", Cost: " + str(loss.data[0]))
    return model


def test(show_metric=False):
    """
    Test the model
    :return: nothing
    """
    #print("Testing")
    model.eval()  # set model to test mode
    total_num_ex = 0
    correct = 0
    for data, target in test_loader:
        # MAKE SURE DATA VAR IS VOLATILE or you get out of mem errors
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total_num_ex += target.size(0)
        correct += (predicted == target.data).sum()
    if show_metric:
        print('Accuracy of the network on the ' + str(total_num_ex) + ' test images: ' +
              str(100 * correct / total_num_ex))
    pass


# to load your previously training model:
# it won’t save the epoch and the optimizer state
model.load_state_dict(torch.load('model.pkl'))
test(show_metric=False)
