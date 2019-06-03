from torch import nn
import torch
from .dis_parts import *

class Dis(nn.Module):
    def __init__(self, in_ch):
        super(Dis, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            res_unit_nb(32, 32),
            down(32, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 512),
            down(512, 512)) # need more down
        self.fc = nn.Sequential(
            nn.Linear(4*4*512, 128), # input channel
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)