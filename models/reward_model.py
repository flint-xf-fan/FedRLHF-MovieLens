print("Importing reward_model.py")

import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x).squeeze(-1)