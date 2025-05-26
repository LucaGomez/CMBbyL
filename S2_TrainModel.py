#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 19:36:31 2025

@author: lgomez
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import random
import pickle

from model_UNet import UNetL
from model_LV3net import CMBFSCNN_lv3_2o
from training_tools import custom_loss
'''Training parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Unt = True
Lv3 = False
batch_size = 2
iteration_n = 100
repeat_n = 2
lr = 1e-3
l_def = 1
l_fft = 1
amount_lr = 1
'''Read the parameters of the simulation from the pickle file generated in S1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
with open('sim_params.pkl', 'rb') as archivo:
    sim_params = pickle.load(archivo)
nside =      sim_params['nside']
freqs =      sim_params['freqs']
dust_model = sim_params['dust']
sync_model = sim_params['sync']
num_train =  sim_params['num_train']
num_valid =  sim_params['num_valid']
r =          sim_params['r']

lmax = 3*nside-1
folder_name = dust_model+sync_model
random.seed(7)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#%%
'''Defines the network architecture.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
if Unt:
    net = UNetL(num_classes=2,in_channels=2*len(freqs),depth=5,start_filts=16,
          up_mode='transpose',merge_mode='concat')
if Lv3:
    net = CMBFSCNN_lv3_2o(in_channels = 2*len(freqs), out_channels = 2, n_feats = 16)

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
net = net.to(device)
print(device)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#%%
'''Define some variables for the training.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay=0)
loss_train = []
loss_t_def = []
loss_t_fft = []
loss_valid = []
loss_v_def = []
loss_v_fft = []
it_pts = np.linspace(0,iteration_n,amount_lr+1,dtype=int)
ch_pts = it_pts[1:-1]
'''Execute the training.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
if not os.path.exists('Results'):
    os.makedirs('Results')
    
best_loss = float('inf')  # <<<
best_model_path = "unet_model_"+str(dust_model)+str(sync_model)+".pth"  # <<<
os.chdir(folder_name)

for iteration in tqdm(range(1, iteration_n+1)):
    for t in range(repeat_n):
        map_nums = np.random.choice(num_train, batch_size, replace=False)
        input_maps = []
        target_maps = []
        facts = []
        for i in range(batch_size):
            tar = []
            data = np.load(str(map_nums[i]) + 'tr.npz')
            box_cont = data['freq_cont_conv_block_qu']
            box_cmb = data['block_cmb_conv_qu']
            fact = data['fact']
            facts.append(fact)
            inp_q = torch.tensor(box_cont[:,0])
            inp_u = torch.tensor(box_cont[:,1])
            inp = torch.cat((inp_q,inp_u), dim=0)
            tar_q = torch.tensor(box_cmb[0])
            tar_u = torch.tensor(box_cmb[1])
            tar = torch.stack((tar_q,tar_u), dim=0)
            input_maps.append(inp)
            target_maps.append(tar)
        input_maps = torch.stack(input_maps, dim=0)
        input_maps = input_maps.to(torch.float32)
        target_maps = torch.stack(target_maps, dim=0)
        input_maps, target_maps = input_maps.to(device), target_maps.to(device)
        
        pred = net(input_maps).squeeze()
        loss_def,loss_fft, = custom_loss(pred, target_maps,l_def, l_fft)
        loss = loss_def+loss_fft
        loss_train.append(loss.item())
        
        loss_t_def.append(loss_def.item())
        loss_t_fft.append(loss_fft.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if iteration in ch_pts:
        lr = lr * 1e-1
        optimizer.param_groups[0]['lr'] = lr
        print('New learning rate = '+str(optimizer.param_groups[0]['lr']))
        
    map_nums = np.random.choice(num_valid, batch_size, replace=False)
    input_maps = []
    target_maps = []
    facts = []
    for i in range(batch_size):
        data = np.load(str(map_nums[i]) + 'tr.npz')
        box_cont = data['freq_cont_conv_block_qu']
        box_cmb = data['block_cmb_conv_qu']
        fact = data['fact']
        facts.append(fact)
        inp_q = torch.tensor(box_cont[:,0])
        inp_u = torch.tensor(box_cont[:,1])
        inp = torch.cat((inp_q,inp_u), dim=0)
        tar_q = torch.tensor(box_cmb[0])
        tar_u = torch.tensor(box_cmb[1])
        tar = torch.stack((tar_q,tar_u), dim=0)
        input_maps.append(inp)
        target_maps.append(tar)
    input_maps = torch.stack(input_maps, dim=0)
    input_maps = input_maps.to(torch.float32)
    target_maps = torch.stack(target_maps, dim=0)
    input_maps,target_maps = input_maps.to(device), target_maps.to(device)
    pred = net(input_maps).squeeze()
    loss_def,loss_fft = custom_loss(pred, target_maps,l_def, l_fft)
    loss_val = loss_def+loss_fft
    
    if loss_val.item() < best_loss:
        best_loss = loss_val.item()
        os.chdir('../Results')
        torch.save(net.state_dict(), best_model_path)
        os.chdir('..')
        os.chdir(folder_name)
    for i in range(repeat_n):
        loss_valid.append(loss_val.item())
    
        loss_v_def.append(loss_def.item())
        loss_v_fft.append(loss_fft.item())
        
print('First loss')
print(loss_train[0])
print('Last loss')
print(loss_train[-1])    

os.chdir('../Results')
plt.figure()
plt.plot(loss_train, label = 'Train')
plt.plot(loss_valid, label = 'Valid')
for k in ch_pts:
    plt.axvline(repeat_n*k, linestyle = 'dotted', color = 'g')
plt.yscale('log')
plt.legend()
plt.savefig('loss_'+str(dust_model)+str(sync_model)+'.png')
plt.show()

plt.figure()
plt.plot(loss_t_def, label = 'Train def')
plt.plot(loss_v_def, label = 'Valid def')
plt.plot(loss_t_fft, label = 'Train fft')
plt.plot(loss_v_fft, label = 'Valid fft')
for k in ch_pts:
    plt.axvline(repeat_n*k, linestyle = 'dotted', color = 'g')
plt.yscale('log')
plt.title('Def wgt = '+str(l_def)+' -- FFT wgt = '+str(l_fft))
plt.legend(fontsize=8)
plt.savefig('component_loss_'+str(dust_model)+str(sync_model)+'.png')
plt.show()
os.chdir('..')
'''Edit the dictionary to save the training parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
sim_params['Unt'] =          Unt
sim_params['Lv3'] =          Lv3
sim_params['batch_size'] =   batch_size
sim_params['interation_n'] = iteration_n
sim_params['repeat_n'] =     repeat_n
sim_params['lr'] =           lr
sim_params['l_def'] =        l_def
sim_params['l_fft'] =        l_fft
sim_params['amount_lr'] =    amount_lr
with open('sim_params.pkl', 'wb') as f:
    pickle.dump(sim_params, f)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
