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

        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 50)
        self.fc4 = nn.Linear(50, 30)
        self.fc5 = nn.Linear(30, 12)
        self.fc6 = nn.Linear(12, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
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

        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 30)
        self.fc4 = nn.Linear(30, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Normalizer():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std
