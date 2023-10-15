import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from scipy.signal import kaiserord, firwin
 
def SAM(I1,I2):
    p1 = np.sum(I1*I2,0)
    p2 = np.sum(I1*I1,0)
    p3 = np.sum(I2*I2,0)
    p4 = np.sqrt(p2*p3)
    p5 = p4.copy()
    p5[p5==0]=1e-15

    sam = np.arccos(p1/p5)
    p1 = p1.ravel()
    p4 = p4.ravel()
    s1 = p1[p4!=0]
    s2 = p4[p4!=0]
    x = (s1/s2)
    x[x>1] = 1
    angolo = np.mean(np.arccos(x))
    sam = np.real(angolo)*180/np.pi
    
    return sam

def ERGAS(I1,I2,c=4):
    s = 0
    R = I1-I2
    for i in range(c):
        res = R[i]
        s += np.mean(res*res)/(np.mean(I1[i])*np.mean(I1[i]))
    s = s/c
    ergas = (100/4) * np.sqrt(s)
    
    return ergas


class Sobel(torch.nn.Module):
    """sobel gradient"""

    def __init__(self,kernel1,kernel2):
        super(Sobel, self).__init__()
        self.g_x = kernel1   
        self.g_y = kernel2

    def forward(self, x):   
        ch = x.shape[1]        
        out_x = F.conv2d(x, self.g_x.repeat(ch,1,1,1), stride=1, padding=1, groups=ch)
        out_y = F.conv2d(x, self.g_y.repeat(ch,1,1,1), stride=1, padding=1, groups=ch)
        return [out_x,out_y]

class Sobel_T(torch.nn.Module):
    """transposed sobel gradient"""

    def __init__(self,kernel1,kernel2):
        super(Sobel_T, self).__init__() 
        self.g_x = kernel1  
        self.g_y = kernel2 

    def forward(self, x):   
        ch = x[0].shape[1]
        out_x = F.conv_transpose2d(x[0], self.g_x.repeat(ch,1,1,1), stride=1, padding=1, groups=ch)
        out_y = F.conv_transpose2d(x[1], self.g_y.repeat(ch,1,1,1), stride=1, padding=1, groups=ch)
        return out_x+out_y
  

 

