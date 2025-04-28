import torch
from torch import nn

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        
    def forward(self, x):
        x = self.flatten(x)
        output = self.network_stack(x)
        return output