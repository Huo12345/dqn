import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):

    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        size = 4
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x
