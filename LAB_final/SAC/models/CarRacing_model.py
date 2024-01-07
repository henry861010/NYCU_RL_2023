import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class ActorNetSimple(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, N_frame: int, game: str, obImage = False) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            #nn.Linear(256, action_dim),
            #nn.LayerNorm(action_dim),
            #nn.Tanh()
        )
        self.mean = nn.Sequential(
            #nn.Linear(256, 256),   v2
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim)
        )
        self.deviation = nn.Sequential(
            #nn.Linear(256, 256),  v2
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim)
        )
        #self.fc1 = torch.nn.Linear(state_dim, 256)
        #self.fc_mu = torch.nn.Linear(256, action_dim)
        #self.fc_std = torch.nn.Linear(256, action_dim)

    def forward(self, state):
        h = self.conv(state)
        h = torch.flatten(h, start_dim=1)
        h = self.linear(h)
        
        means = self.mean(h)  # prevent the value smaller than 1e-7
        deviations = self.deviation(h).exp()   # the deviation should be non-negative

        dist = Normal(means, deviations)
        action = dist.rsample()

        log_prob = dist.log_prob(action)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        action[:, 0] = torch.sigmoid(action[:, 0])*1.5-0.5  # 1~-0.4 v2

        #action[:, 0] = torch.sigmoid(action[:, 0])  # 0~1  v1
        action[:, 1] = torch.tanh(action[:, 1])  # -1~1
         #action[:, 0] = torch.tanh(action[:, 0])  # -1~1
        #action[:, 1] = torch.sigmoid(action[:, 1])  # 0~1
        


        #x = F.relu(self.fc1(state))
        #mu = self.fc_mu(x)
        #std = F.softplus(self.fc_std(x))
        #dist = Normal(mu, std)
        #normal_sample = dist.rsample()  # rsample()是重参数化采样
        #log_prob = dist.log_prob(normal_sample)
        #action = torch.tanh(normal_sample)
        ### 计算tanh_normal分布的对数概率密度
        #log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        #action = action * 2

        return action, log_prob

    
class CriticNetSimple(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, N_frame: int, game: str, obImage = False) -> None:
        super().__init__()
        self.game = game
        self.obImage = obImage

        self.conv = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.ELU()
        )

        self.state_linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )

        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        #self.fc1 = torch.nn.Linear(state_dim + action_dim, 256)
        #self.fc2 = torch.nn.Linear(256, 256)
        #self.fc_out = torch.nn.Linear(256, 1)

    def forward(self, state, action):

        state_h = self.conv(state)
        state_h = torch.flatten(state_h, start_dim=1)
        state_h = self.state_linear(state_h)
        action_h = self.action_linear(action)
        h = self.concat_linear(torch.concat((state_h, action_h), dim=1))
            
        #h = torch.cat([state,action], dim=1)
        #h = F.relu(self.fc1(h))
        #h = F.relu(self.fc2(h))
        #h = self.fc_out(h)

        return h
