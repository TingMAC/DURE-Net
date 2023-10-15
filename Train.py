#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of training code of this paper:
# Deep Algorithm Unrolling with Registration Embedding for Pansharpening, ACM MM'23
# author: Tingting Wang, Yongxu Ye, Faming Fang, Guixu Zhang, Ming Xu 
"""
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import h5py
from skimage.metrics import peak_signal_noise_ratio as PSNR
from Model import *
from Utils import *
from Dataset import *
import sys
import importlib
importlib.reload(sys)
import scipy.io as sio
torch.cuda.manual_seed(1)

#devicesList = [0]
dtype = torch.cuda.FloatTensor
MAEloss = torch.nn.L1Loss().type(dtype)
psnr_best = 0

savenet_name = 'YOUR MODEL PATH.pth'
train_data_name = 'YOUR TRAIN DATA PATH.h5'
validation_data_name  = 'YOUR VALIDATION DATA PATH.h5'
 
def validation(dataloader):
    sum_psnr = 0
    sum_sam = 0
    sum_ergas = 0
    count = 0
    CNN.eval()
    for index, data in enumerate(dataloader):
        #count += data[0].shape[0]
        count += 1
        gtVar = Variable(data[3]).type(dtype) 
        panVar = Variable(data[0]).type(dtype) 
        msVar = Variable(data[1]).type(dtype)           
        output,_,_ =  CNN(msVar,panVar)      
        netOutput_np = output.cpu().data.numpy()[0]
        gtLabel_np = gtVar.cpu().data.numpy()[0]
        samValue = SAM(gtLabel_np,netOutput_np)
        ergasValue = ERGAS(gtLabel_np,netOutput_np)        
        psnrValue = PSNR(gtLabel_np,netOutput_np)   
        sum_sam += samValue
        sum_psnr += psnrValue
        sum_ergas += ergasValue       
 
    avg_psnr = sum_psnr/count
    avg_sam = sum_sam/count
    avg_ergas = sum_ergas/count   
    
    print('psnr:%.4f sam:%.4f ergas:%.4f'%(avg_psnr,avg_sam,avg_ergas))
    return avg_psnr,avg_sam,avg_ergas

if __name__ == "__main__":
    ## parameters setting and network selection ##
    train_bs = 32
    val_bs = 1
    test_bs = 1
    epoch = 500
    stages = 4
    Ch = 4  
    nc = 32      
    LR = 0.001
    omega1 = 0.1
    omega2 = 0.1
    #params = torch.tensor([1., 0.01, 1.0, 1.0, 0.1])  # alpha  gamma  mu1  mu2 epsilon ###########    
    CNN = DURE(Ch, stages, nc)
    CNN = nn.DataParallel(CNN).cuda()  
    ## parameters setting ##


    ### read dataset ###       
    train_data = h5py.File(train_data_name,'r')
    train_dataset = my_dataset(train_data)
    trainsetSize = train_data['gt'].shape[0]
    del train_data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    
    validation_data = h5py.File(validation_data_name,'r')
    validation_dataset = my_dataset(validation_data)
    del validation_data  
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=val_bs, shuffle=True)
    
    
    optimizer = torch.optim.Adam(CNN.parameters(), lr=LR, betas=(0.9, 0.999),weight_decay=0.0001)
    
    for i in range(1,epoch+1):          
        count = 0     
        CNN.train()
        for index, data in enumerate(train_dataloader):    
            count += data[0].shape[0]
            optimizer.zero_grad()            
            gtVar = Variable(data[3]).type(dtype) 
            panVar = Variable(data[0]).type(dtype)            
            msVar = Variable(data[1]).type(dtype)      
                 
            img_predict,ms_dual, pan_dual = CNN(msVar,panVar) 
            loss_HRMS = MAEloss(img_predict,gtVar)
            loss_LRMS = MAEloss(ms_dual,msVar)
            loss_PAN = MAEloss(pan_dual,panVar)
            loss = 1000*(loss_HRMS+omega1*loss_LRMS+omega2*loss_PAN  )    
            loss.backward()            
            optimizer.step()
            print('epoch:%04d [%04d/%04d] loss:%.4f HRMS:%.4f LRMS:%.4f PAN:%.4f'%(i,count,trainsetSize,loss.data,loss_HRMS.data,loss_LRMS.data,loss_PAN.data), '\r',end = '\r')
            #print('epoch:%04d [%05d/%05d] loss %.8f '%(i,count,trainsetSize,loss.data), '\r',end = '\r')
            
            
        if (i)%50 == 0:
            LR = LR/2
            optimizer = torch.optim.Adam(CNN.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=0.0001)
                    
        if (i)%2 == 0:            
            print("")  
            print("val:",end='')       
            psnr,sam,ergas = validation(validation_dataloader)
            if psnr > psnr_best:
               torch.save(CNN.state_dict(),savenet_name) 
               psnr_best = psnr
               best_index = i
               print("Best PSNR and epoch:",end='')
               print(psnr_best,best_index) 
               #test(test_full_dataloader,savemat_full_name)     
               #test(test_reduced_dataloader,savemat_reduced_name)     
