import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):

    # timestep of day
    # soe
    # last raise clearing price
    # prediction of demand
    # clearing energy price
    # last cleared energy price maybeee
    # last week same day clearing raise price
    # yesterday same timestep clearing raise price
    # bids from other people - maybe later
    # 2 artificial time-dependent features.

    def __init__(self):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CriticNet(nn.Module):

    # timestep of day
    # soe
    # last raise clearing price
    # prediction of demand
    # clearing energy price
    # last cleared energy price maybeee
    # last week same day clearing raise price
    # yesterday same timestep clearing raise price
    # bids from other people - maybe later
    # 2 artificial time-dependent features.

    def __init__(self):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
