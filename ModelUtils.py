import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
from Utils import *
  
 
class UNet(nn.Module):
    def __init__(self,Ch,nc):
        super(UNet, self).__init__()
        self.upMode = 'bilinear'
        self.conv1 = nn.Conv2d(in_channels=Ch, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nc, out_channels=Ch, kernel_size=3, stride=1, padding=1)
                                                      
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),                                
                                )
        self.E2 = nn.Sequential(nn.Conv2d(in_channels=nc, out_channels=2*nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E3 = nn.Sequential(nn.Conv2d(in_channels=2*nc, out_channels=4*nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E4 = nn.Sequential(nn.Conv2d(in_channels=4*nc, out_channels=4*nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=8*nc, out_channels=2*nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=4*nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=2*nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
      

    def forward(self, x):
        ## encoding blocks
        E1 = self.conv1(x)
        E2 = self.E1(E1)
        E3 = self.E2(F.avg_pool2d(E2, kernel_size=2, stride=2))
        E4 = self.E3(F.avg_pool2d(E3, kernel_size=2, stride=2))
        E5 = self.E4(F.avg_pool2d(E4, kernel_size=2, stride=2))
        sz = E4.size()
        D1 = self.D1(torch.cat([E4, F.interpolate(E5, size=[sz[2],sz[3]], mode=self.upMode)], dim=1))
        sz = E3.size()
        D2 = self.D2(torch.cat([E3, F.interpolate(D1, size=[sz[2],sz[3]], mode=self.upMode)], dim=1))
        sz = E2.size()
        D3 = self.D3(torch.cat([E2, F.interpolate(D2, size=[sz[2],sz[3]], mode=self.upMode)], dim=1))
        D4 = self.conv2(D3)
        return D4
    
     