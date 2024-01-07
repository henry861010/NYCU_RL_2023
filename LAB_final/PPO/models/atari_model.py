import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, N_frame: int, init_weights=True) -> None:
        super(AtariNet, self).__init__()


        self.cnn = nn.Sequential(
                nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
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

        self.action_logits = nn.Sequential(
                nn.Linear(16*(state_dim//8)**2, 256),
                nn.ReLU(True),
                nn.Linear(256, action_dim)
        )

        self.value = nn.Sequential(
                nn.Linear(16*(state_dim//8)**2, 256),
                nn.ReLU(True),
                nn.Linear(256, 1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False):
        #print("***",x.shape)
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)

        value = self.value(x)
        value = torch.squeeze(value)

        logits = self.action_logits(x)
        act_dist = Categorical(logits=logits)
        action = act_dist.sample()
        entropy = act_dist.entropy()
        return action, act_dist, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                


