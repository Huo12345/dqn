import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):

    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 16).double()
        self.fc2 = nn.Linear(16, 2).double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.softmax(self.fc2(x), dim=1)
        return x
