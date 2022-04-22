import torch
import torch.nn as nn

class Separator(nn.Module):
    def __init__(self, window_size):
        super(Separator, self).__init__()
        
        if window_size == 1:
            conv_kernel_size = 2
            linear_channel = 1
        else:
            conv_kernel_size = window_size*2-2
            linear_channel = 3
        
        # block1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=256, kernel_size=conv_kernel_size),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=linear_channel),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # block2
        self.block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Linear Layer --> Logit
        self.linear_block = nn.Sequential(
            nn.Linear(32, 1),
        )
        
    def forward(self, x):        
        out = self.block1(x)
        batch_size = out.size()[0]
        out = out.view(batch_size, -1)
        out = self.block2(out)
        out = self.linear_block(out)
        return out.view(-1)