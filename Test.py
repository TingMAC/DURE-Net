#!/usr/bin/env python2
# -*- coding: utf-8 -*-
 
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.autograd import Variable
from Model import *
from Utils import *
from Dataset import *
import os
import sys
import importlib
importlib.reload(sys)
import scipy.io as sio

 
dtype = torch.cuda.FloatTensor
nc = 32
stages = 4
Ch = 4  ## the channel number of MS images 
#params = torch.tensor([1., 0.01, 1.0, 1.0, 0.1])  # alpha  gamma  mu1  mu2 epsilon ###########
 
if __name__ == "__main__":
    ##### read dataset #####
    test_bs = 1    
    tmpPath = "YOUR MODEL PATH.pth"    
    test_data_name  = 'YOUR TEST DATA PATH.h5'
   
   
    test_data = h5py.File(test_data_name,'r')
    test_dataset = my_dataset(test_data)
    del test_data
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False) 
   
    CNN = DURE(Ch, stages, nc)  ## others
    CNN = nn.DataParallel(CNN).cuda()  
    CNN.load_state_dict(torch.load(tmpPath))
    CNN.eval()
    count = 0    
 
    for _, data in enumerate(test_dataloader):
        #count += data[0].shape[0]            
        msVar = Variable(data[1]).type(dtype)
        panVar = Variable(data[0]).type(dtype)               
        output,_,_ = CNN(msVar,panVar)      
        netOutput_np = output.cpu().data.numpy()
        if count == 0:
            Output_np = netOutput_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
        count = count + 1
    
    sio.savemat(SaveDataPath,{'sr':Output_np}) 
            