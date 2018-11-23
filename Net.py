import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy

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
        return F.log_softmax(x, dim=1)


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
        return F.log_softmax(x, dim=1)
