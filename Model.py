import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from Utils import *
from ModelUtils import *
import pdb


def setup_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seeds(1)


class DURE(nn.Module): ## without alpha, with two thr
    def __init__(self, Ch = 4, stages = 4, nc = 32):
        super(DURE, self).__init__()
        self.s  = stages
        self.upMode = 'bilinear'
        #self.mu = params[0]
        self.nc = nc
        self.DNet = UNet(Ch, self.nc)        
        sobel_x = (torch.FloatTensor([[-1.0,0,-1.0],[-2.0,0,2.0],[-1.0,0,-1.0]])).cuda()
        sobel_x = sobel_x.unsqueeze(dim=0).unsqueeze(dim=0)
        sobel_y = (torch.FloatTensor([[-1.0,-2.0,-1.0],[0,0,0],[1.0,2.0,1.0]])).cuda()
        sobel_y = sobel_y.unsqueeze(dim=0).unsqueeze(dim=0)
        self.sobel = Sobel(sobel_x,sobel_y)
        self.Tsobel = Sobel_T(sobel_x,sobel_y)
        ## The modules for learning the measurement matrix D and D^T
        self.DT = nn.Sequential(nn.ConvTranspose2d(Ch, self.nc, kernel_size=3, stride=2, padding=1,output_padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.ConvTranspose2d(self.nc, Ch, kernel_size=3, stride=2, padding=1,output_padding=1))
        self.D  = nn.Sequential(nn.Conv2d(Ch, self.nc, kernel_size=3, stride=2, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, Ch, kernel_size=3, stride=2, padding=1))


        ## The modules for learning the measurement matrix G and G^T
        self.GT = nn.Sequential(nn.Conv2d(1, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, Ch, kernel_size=3, stride=1, padding=1))
        self.G  = nn.Sequential(nn.Conv2d(Ch, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(self.nc, 1, kernel_size=3, stride=1, padding=1))    

        self.H_step = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.U_step = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.V_step = Parameter(0.1*torch.ones(self.s, 1),requires_grad=True)
        self.mu = Parameter(torch.ones(1), requires_grad=True)
        #self.alpha = Parameter(torch.ones(1), requires_grad=True)
        self.soft_thr1 = Parameter(0.01*torch.ones(1), requires_grad=True)   
        self.soft_thr2 = Parameter(0.01*torch.ones(1), requires_grad=True)   
        self._initialize_weights()
        torch.nn.init.normal_(self.H_step, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.U_step, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.V_step, mean=0.1, std=0.01)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)  

    def forward(self, M,P):   
        [P_x,P_y] = self.sobel(P)  
        Ht = F.interpolate(M , scale_factor = 4, mode = self.upMode)
        U = torch.zeros(P.shape).cuda()
        V = torch.zeros(P.shape).cuda()
        B1x = torch.zeros(P.shape).cuda()
        B1y = torch.zeros(P.shape).cuda()
        B2x = torch.zeros(P.shape).cuda()
        B2y = torch.zeros(P.shape).cuda()
         
        for i in range(0, self.s):           

            ## U subproblem  
            [U_x,U_y] = self.sobel(U)  
            Grad_U = P_x*(P_x*U+P_y*V+P-self.G(Ht))+self.mu*self.Tsobel([U_x-B1x,U_y-B1y])
            U = U - self.U_step[i]  * Grad_U
            [U_x,U_y] = self.sobel(U) 
            B1x = torch.mul(torch.sign(U_x), F.relu(torch.abs(U_x) - self.soft_thr1)) 
            B1y = torch.mul(torch.sign(U_y), F.relu(torch.abs(U_y) - self.soft_thr1))             
         
            ## V subproblem 
            [V_x,V_y] = self.sobel(V)   
            Grad_V = P_y*(P_x*U+P_y*V+P-self.G(Ht))+self.mu*self.Tsobel([V_x-B2x,V_y-B2y])
            V = V - self.V_step[i]  * Grad_V          
            [V_x,V_y] = self.sobel(V) 
            B2x = torch.mul(torch.sign(V_x), F.relu(torch.abs(V_x) - self.soft_thr2)) 
            B2y = torch.mul(torch.sign(V_y), F.relu(torch.abs(V_y) - self.soft_thr2)) 

            ## Reconstructing HRMS                         
            Grad_H = self.GT(self.G(Ht)-P-(P_x*U+P_y*V))+self.DT(self.D(Ht)-M)         
            Ht = Ht - self.H_step[i]  * Grad_H
            Ht = Ht+self.DNet(Ht)
            
        ## consistency loss
        M_dual = self.D(Ht)
        P_dual = self.G(Ht)-P_x*U-P_y*V
        return Ht,M_dual,P_dual