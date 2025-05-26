#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:02:55 2025

@author: lgomez
"""

import numpy as np
import healpy as hp
import pymaster as nmt
import os
import pickle
import random
from tqdm import tqdm

from nah_master import nah_master
from spherical import blockPlane2sphere_mult
from simulation import Cls_gen,params,coverage_map,reduct_tools_from_coverage,frequency_blocks_test,block_upgrade

'''Read the parameters of the simulation from the pickle file generated in S1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
with open('sim_params.pkl', 'rb') as archivo:
    sim_params = pickle.load(archivo)
nside = sim_params['nside']
freqs = sim_params['freqs']
out_freq_i = sim_params['out_freq_i']
beams = sim_params['beams']
sens = sim_params['sens']
dust_model = sim_params['dust']
sync_model = sim_params['sync']
num_train = sim_params['num_train']
num_valid = sim_params['num_valid']
r = sim_params['r']
nlb = sim_params['nlb']
ap = sim_params['ap']
cent_vec = sim_params['cent_vec']
rad_std = sim_params['rad_std']
rad_max = sim_params['rad_max']
rad_num = sim_params['rad_num']
max_hit = sim_params['max_hit']

num_test = 10
lmax = 3*nside-1
folder_name = dust_model+sync_model
random.seed(7)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

#%%
'''Defines the mask with the apodization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
pars = params()
Cls, ell = Cls_gen(pars, r, lmax)
coverage = coverage_map(nside,cent_vec,rad_std,rad_max,rad_num,max_hit)
tools = reduct_tools_from_coverage(coverage)
mask = np.ones(hp.nside2npix(nside))
mask[coverage==hp.UNSEEN] = 0
mask = nmt.mask_apodization(mask, aposize=ap, apotype="Smooth")
np.save('mask.npy',mask)
np.save('tools.npy',tools)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
#%%
'''Generates the folder to work during the Testing Stage.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
os.chdir('Results')
if not os.path.exists('Test_set'):
    os.makedirs('Test_set')
os.chdir('Test_set')
for j in tqdm(range(num_test)):
    real = []
    box_cont, box_cmb, fact = frequency_blocks_test(nside,freqs,out_freq_i,beams,sens,coverage,
                                                    tools,dust_model,sync_model,lmax,Cls)
    ell = np.arange(lmax+1)
    
    tar_q = box_cmb[0]*fact
    tar_u = box_cmb[1]*fact
    tar = np.array([tar_q,tar_u])
    
    cmb_block = block_upgrade(box_cmb*fact, tools, nside)
    
    sp_tar_q = blockPlane2sphere_mult(cmb_block[0], nside=nside, block_n='block_11')
    sp_tar_u = blockPlane2sphere_mult(cmb_block[1], nside=nside, block_n='block_11')
    
    mp_qu_tar = [sp_tar_q,sp_tar_u]

    ll, clsn_tar = nah_master(mp_qu_tar, mask, bl=None, nside=nside, nlb=nlb)

    clee_tar = clsn_tar[0]
    clbb_tar = clsn_tar[3]
    
    np.savez(str(j) +'te.npz', box_cont=box_cont, box_cmb=box_cmb, fact=fact, ell=ell,
             ll=ll, clee_tar=clee_tar, clbb_tar=clbb_tar)
    
os.chdir('../..')
'''Edit the dictionary to save the training parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
sim_params['num_test'] = num_test
with open('sim_params.pkl', 'wb') as f:
    pickle.dump(sim_params, f)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''