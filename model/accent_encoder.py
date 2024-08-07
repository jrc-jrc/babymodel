# acoustic_encoder.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same')
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class AcousticEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=256):
        super(AcousticEncoder, self).__init__()
        self.convs = nn.Sequential(
            ConvBlock(input_dim, 512, 5, 1),
            ConvBlock(512, 512, 3, 2),
            ConvBlock(512, 512, 3, 1),
            ConvBlock(512, output_dim, 1, 1)
        )
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.convs(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.attention(x, x, x)
        return self.pool(x.transpose(1, 2)).squeeze(-1)