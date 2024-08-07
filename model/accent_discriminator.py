# accent_discriminator.py
import torch
import torch.nn as nn

class AccentDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(AccentDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))