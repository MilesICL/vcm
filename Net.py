import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy

class NetVCM(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(NetVCM, self).__init__()
        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nHidden)
        self.fc3 = nn.Linear(nHidden, nHidden)
        self.fc4 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


class NetLing(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(NetLing, self).__init__()
        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nHidden)
        self.fc3 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class NetSyll(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(NetSyll, self).__init__()
        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nHidden)
        self.fc3 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
