#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:23:59 2025

@author: lgomez
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import pickle
import torch

from simulation import block_upgrade
from model_UNet import UNetL
from model_LV3net import CMBFSCNN_lv3_2o
from nah_master import nah_master
from spherical import blockPlane2sphere_mult
'''Read the parameters of the simulation from the pickle file generated in S1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
with open('sim_params.pkl', 'rb') as archivo:
    sim_params = pickle.load(archivo)
nside = sim_params['nside']
freqs = sim_params['freqs']
out_freq_i = sim_params['out_freq_i']
beams = sim_params['beams']
dust_model = sim_params['dust']
sync_model = sim_params['sync']
nlb = sim_params['nlb']
num_train = sim_params['num_train']
num_valid = sim_params['num_valid']
Unt = sim_params['Unt']
Lv3 = sim_params['Lv3']
r = sim_params['r']
num_test = sim_params['num_test']

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
mask = np.load('mask.npy')
tools = np.load('tools.npy')
#%%
'''This part can be used to test a network without the training
'''
os.chdir('Results')
model_path = "unet_model_"+str(dust_model)+str(sync_model)+".pth"
net.load_state_dict(torch.load(model_path))
net.eval()
'''
'''
#%%
if not os.path.exists('Rec_maps'):
    os.makedirs('Rec_maps')
if not os.path.exists('Cls_est'):
    os.makedirs('Cls_est')
#%%
map_Q_mean = []
map_Q_max = []
map_U_mean = []
map_U_max = []

EE_dif_spec = []
BB_dif_spec = []

for j in tqdm(range(num_test)):
    os.chdir('Test_set')
    real = []
    data = np.load(str(j) + 'te.npz')
    box_cont = data['box_cont']
    box_cmb = data['box_cmb']
    fact = data['fact']
    ell = data['ell']
    ll = data['ll'] 
    clee_tar = data['clee_tar'] 
    clbb_tar = data['clbb_tar']
    
    inp_q = torch.tensor(box_cont[:,0])
    inp_u = torch.tensor(box_cont[:,1])
    inp = torch.cat((inp_q,inp_u), dim=0).unsqueeze(0).to(torch.float32)
    tar_q = box_cmb[0]*fact
    tar_u = box_cmb[1]*fact
    tar = np.array([tar_q,tar_u])

    inp = inp.to(device)
    pred = net(inp).squeeze()
    rec = pred.cpu().detach().numpy()*fact
    
    res = rec - tar
    
    map_Q_mean.append(np.mean(np.abs(res[0])))
    map_Q_max.append(np.max(np.abs(res[0])))
    map_U_mean.append(np.mean(np.abs(res[1])))
    map_U_max.append(np.max(np.abs(res[1])))
    
    os.chdir('../Rec_maps')

    images = []
    images.append(rec[0])
    images.append(tar[0])
    images.append(res[0])
    images.append(rec[1])
    images.append(tar[1])
    images.append(res[1])
    

    
    names = ['Q Predicted', 'Q Target', 'Q Res', 
             'U Predicted', 'U Target', 'U Res']
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # Iterar sobre los ejes y las imágenes
    for i, (ax, img) in enumerate(zip(axes.ravel(), images), 1):
        im = ax.imshow(img, cmap='viridis')  # Mostrar imagen
        ax.set_title(names[i-1])  # Agregar título
        ax.axis('off')  # Ocultar ejes
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Agregar barra de color
        
    plt.tight_layout()  # Ajustar espacios
    plt.savefig('Maps_'+str(j)+'.png')
    plt.show()
    plt.close()
    
    os.chdir('..')
    
    cmb_block = block_upgrade(rec, tools, nside)
    
    sp_rec_q = blockPlane2sphere_mult(cmb_block[0], nside=nside, block_n='block_11')
    sp_rec_u = blockPlane2sphere_mult(cmb_block[1], nside=nside, block_n='block_11')
    
    mp_qu_rec = [sp_rec_q,sp_rec_u]
    
    os.chdir('Cls_est')

    ll, clsn_rec = nah_master(mp_qu_rec, mask, bl=None, nside=nside, nlb=nlb)
    
    clee_rec = clsn_rec[0]
    clbb_rec = clsn_rec[3]
    
    dif_E = 100*np.abs(clee_rec-clee_tar)/np.abs(clee_tar)
    dif_B = 100*np.abs(clbb_rec-clbb_tar)/np.abs(clbb_tar)
    
    EE_dif_spec.append(dif_E)
    BB_dif_spec.append(dif_B)
    
    # Crear figura con 1 fila y 2 columnas
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Graficar la primera función
    #axes[0].plot(ell,ClEE, label = 'EE camb')
    axes[0].scatter(ll, clee_tar, marker = '.', color = 'g', label = 'EE tar')
    axes[0].scatter(ll, clee_rec, marker = '.', color = 'r', label = 'EE rec')
    axes[0].set_title("cls -- r = "+str(r))
    axes[0].set_xlabel(r'$\ell$')
    axes[0].set_ylabel(r'$C_\ell^{EE}(\mu K^2)$')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_xlim(40,lmax)
    axes[0].legend()
    axes[0].grid(True)

    # Graficar la segunda función
    axes[1].scatter(ll,dif_E, marker = '.')
    axes[1].set_title(r"% diff: $C_\ell^{\rm EE-tar}--C_\ell^{\rm EE-rec}$")
    axes[1].set_xlabel(r'$\ell$')
    axes[1].set_ylabel(r'err%($C_\ell^{\rm EE})$')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlim(40,lmax)
    axes[1].grid(True)

    plt.tight_layout()  # Ajustar espacios
    plt.savefig('Cls_E_'+str(j)+'.png')
    plt.show()
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Graficar la primera función
   #axes[0].plot(ell,ClBB, label = 'BB camb')
    axes[0].scatter(ll, clbb_tar, marker = '.', color = 'g', label = 'BB tar')
    axes[0].scatter(ll, clbb_rec, marker = '.', color = 'r', label = 'BB rec')
    axes[0].set_title("cls -- r = "+str(r))
    axes[0].set_xlabel(r'$\ell$')
    axes[0].set_ylabel(r'$C_\ell^{BB}(\mu K^2)$')
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].set_xlim(40,lmax)
    axes[0].legend()
    axes[0].grid(True)

    # Graficar la segunda función
    axes[1].scatter(ll,dif_B, marker = '.')
    axes[1].set_title(r"% diff: $C_\ell^{\rm BB-tar}--C_\ell^{\rm BB-rec}$")
    axes[1].set_xlabel(r'$\ell$')
    axes[1].set_ylabel(r'err%($C_\ell^{BB})$')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlim(40,lmax)
    axes[1].grid(True)

    plt.tight_layout()  # Ajustar espacios
    plt.savefig('Cls_B_'+str(j)+'.png')
    plt.show()
    plt.close()
    
    os.chdir('..')
print('Mean Q residual averaged over test set')
print(np.mean(map_Q_mean))
#print(np.std(map_Q_mean)/np.sqrt(num_test))
print('Max Q residual averaged over test set')
print(np.mean(map_Q_max))
#print(np.std(map_Q_max)/np.sqrt(num_test))
print('Mean U residual averaged over test set')
print(np.mean(map_U_mean))
#print(np.std(map_U_mean)/np.sqrt(num_test))
print('Max U residual averaged over test set')
print(np.mean(map_U_max))
#print(np.std(map_U_max)/np.sqrt(num_test))

EE_dif_spec = np.array(EE_dif_spec) 
BB_dif_spec = np.array(BB_dif_spec)

EE_dif_mean = np.mean(EE_dif_spec, axis=0)
EE_dif_err = np.std(EE_dif_spec, axis=0) / np.sqrt(num_test)
BB_dif_mean = np.mean(BB_dif_spec, axis=0)
BB_dif_err = np.std(BB_dif_spec, axis=0) / np.sqrt(num_test)

plt.errorbar(
    ll, EE_dif_mean, 
    yerr=EE_dif_err,  # Barras de error
    fmt='o',               # Formato del marcador (círculo)
    markersize=.5,          # Tamaño del marcador
    capsize=5,             # Tamaño de las tapas de error
    capthick=1.5,          # Grosor de las tapas de error
    ecolor='red',          # Color de las barras de error
    markerfacecolor='blue', # Color interno del marcador
    markeredgecolor='black', # Color del borde del marcador
    linestyle='None'       # Sin líneas conectando puntos
)
plt.title('EE')
plt.xlabel(r'$\ell$', fontsize=12)
plt.ylabel('<%diff>', fontsize=12)
plt.yscale('log')
plt.xlim(40,lmax)

# Ajustar el estilo del grid
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('EE_avg.png')

# Mostrar el gráfico
plt.show()


plt.errorbar(
    ll, BB_dif_mean, 
    yerr=BB_dif_err,  # Barras de error
    fmt='o',               # Formato del marcador (círculo)
    markersize=.5,          # Tamaño del marcador
    capsize=5,             # Tamaño de las tapas de error
    capthick=1.5,          # Grosor de las tapas de error
    ecolor='red',          # Color de las barras de error
    markerfacecolor='blue', # Color interno del marcador
    markeredgecolor='black', # Color del borde del marcador
    linestyle='None'       # Sin líneas conectando puntos
)
plt.title('BB')
plt.xlabel(r'$\ell$', fontsize=12)
plt.ylabel('<%diff>', fontsize=12)
plt.yscale('log')
plt.xlim(40,lmax)

# Ajustar el estilo del grid
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('BB_avg.png')
# Mostrar el gráfico
plt.show()
os.chdir('..')

