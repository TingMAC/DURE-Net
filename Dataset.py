#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import torch.utils.data as data
import numpy as np
import torch
import matplotlib.pyplot as plt
#import spectral as spy
import torch.nn as nn

class my_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, ms, lms,gt

    def __len__(self):
        return self.gt_set.shape[0]

class my_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, ms, lms

    def __len__(self):
        return self.pan_set.shape[0]


if __name__ == "__main__":
    validation_data_name = 'YOUR PATH.h5'  # your data path
    validation_data = h5py.File(validation_data_name, 'r')
    validation_dataset = my_dataset(validation_data)
    del validation_data
    data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)
    for index, item in enumerate(data_loader):
        print(item[0])      
        plt.show()
        if index==1:break
