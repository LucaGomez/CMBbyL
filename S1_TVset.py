#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:12:23 2025
@author: lgomez

This code implement the functions of -simulation.py- to generate the inputs and the targets
for the CNN training. The scheme of work of this code is as follow:
    cosmological_params ---camb---> Cls[TT,EE,BB,TE,PP,PT,PE]     (1)
    Cls ----------------lenspyx---> Lensed CMB realization        (2)
    Foreg. model+Freqs ---pysm3---> Frequency foreground maps     (3)
    (2)+(3)+Beams -------healpy---> Frequency contaminated maps   (4)
    (4) -----------spherical.py---> Frequency contaminated blocks (5)
    (5) ----------simulation.py---> Inputs,target CNN             (6)
Update: Noise included with the coverage, and also the smaller-blocks strategy.
"""
import numpy as np
import healpy as hp
import pickle
import random

from simulation import params,Cls_gen,coverage_map,reduct_tools_from_coverage,sets_generator
'''Simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
nside = 512
freqs = np.array([30,85,145,220,270])         # GHz
beams = np.array([7.3,2.3,1.5,1,0.8])/60      # Deg        (np.radians() in conv_with_beam())
sens = np.array([3.53,0.88,1.23,3.48,5.97])   # muK*arcmin (sens**2/Om_pix_amin2 in noise_raw_map())
out_freq_i = 3
dust_model = 'd4'
sync_model = 's2'
num_train = 20
num_valid = 5
random.seed(7)
r = 0
lmax = 3 * nside - 1
rad_std = 6
rad_max = 14
rad_num = 50
cent_vec = hp.ang2vec(2.548, 5.515)
cent = (316, -56)
max_hit = 40
ap = 0.5
nlb = 12
'''Import the cosmological parameters from -simulation.py- and compute the Cls dictionary.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
pars = params()
Cls, ell = Cls_gen(pars, r, lmax)
coverage = coverage_map(nside,cent_vec,rad_std,rad_max,rad_num,max_hit)
tools = reduct_tools_from_coverage(coverage)
'''Generate the train and valid test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
sets_generator(num_train,num_valid,nside,freqs,out_freq_i,beams,sens,coverage,tools,dust_model,
               sync_model,Cls,lmax,Train=True,Valid=True)
'''Generate the dictionary to save the simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
efect_nside = tools[-1]
sim_params = {
    'nside': nside,
    'efect_nside': efect_nside,
    'freqs': freqs,
    'out_freq_i': out_freq_i,
    'beams': beams,
    'sens': sens,
    'dust': dust_model,
    'sync': sync_model,
    'num_train': num_train,
    'num_valid': num_valid,
    'r': r,
    'rad_std': rad_std,
    'rad_max': rad_max,
    'rad_num': rad_num,
    'cent_vec': cent_vec,
    'cent': cent,
    'max_hit': max_hit,
    'ap': ap,
    'nlb': nlb}
with open('sim_params.pkl', 'wb') as archivo:
    pickle.dump(sim_params, archivo)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

