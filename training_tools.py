#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:58:13 2025

@author: lgomez
"""

import torch

def custom_loss(cmb_fals, cmb_true, l_def, l_fft):
    
    loss_func = torch.nn.L1Loss()
    loss_def = loss_func(cmb_fals,cmb_true)
    
    if l_fft:
        loss_fft = fft_loss(cmb_fals, cmb_true)
    else:
        loss_fft = torch.zeros_like(loss_def)
        
    return l_def * loss_def, l_fft * loss_fft

def fft_loss(cmb_false, cmb_true):
    loss_func = torch.nn.L1Loss()
    cmb_false_fft = torch.abs(torch.fft.fftn(cmb_false, dim=(2, 3))) / cmb_true.shape[3]
    cmb_true_fft = torch.abs(torch.fft.fftn(cmb_true, dim=(2, 3))) / cmb_true.shape[3]
    return loss_func(cmb_true_fft, cmb_false_fft)

