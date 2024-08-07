# hifigan_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HiFiGANDecoder(nn.Module):
    def __init__(self, input_dim, upsample_rates, upsample_kernel_sizes):
        super(HiFiGANDecoder, self).__init__()
        self.conv_pre = nn.Conv1d(input_dim, 512, kernel_size=11, stride=1, padding=5)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(512 // (2**i), 512 // (2**(i+1)),
                                               k, u, padding=(k-u)//2))

        self.conv_post = nn.Conv1d(512 // (2**len(upsample_rates)), 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x