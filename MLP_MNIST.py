import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

try:
    import matplotlib.pyplot as plt
    from tensorboardX import SummaryWriter
    tb = True
except ModuleNotFoundError:
    tb = False

# TODO: Import data here:

kwargs = {}
# creates a data loader - downloads it into ./data and transforms it to pytorch tensor and normalizes it to gaussian
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=20, shuffle=True, **kwargs)

print(train_loader)

##

# TODO: Define your model:
class Net(nn.Module):
    def __init__(self, n_nodes, n_classes):
        super(Net, self).__init__()
        # define the architecture
        self.fc1 = nn.Linear(28*28, )
        self.fc2 = nn.Linear(n_nodes[0], n_nodes[1]) # hl1
        self.fc3 = nn.Linear(n_nodes[1], n_nodes[2]) # hl2
        self.fc4 = nn.Linear(n_nodes[1], n_nodes[2]) # hl3
        self.output = nn.Linear(n_nodes[2], n_classes)

        # Define layers making use of torch.nn functions:

    def forward(self, x):
        x = F.relu()
        pass

        # Define how forward pass / inference is done:

        # return out #return output


my_net = Net()

# TODO: Train your model:

# torch.save(my net.state dict(), ’model.pkl’)
