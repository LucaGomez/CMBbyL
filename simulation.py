#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:58:02 2025

@author: lgomez
"""
import numpy as np
import pysm3
import pysm3.units as u
import os
from tqdm import tqdm
import camb
import healpy as hp
from lenspyx import synfast

from spherical import sphere2piecePlane, piecePlane2blocks

H0_m = 67.4
H0_e = 0.5

ombh2_m = 0.0224
ombh2_e = 0.0001

omch2_m = 0.120
omch2_e = 0.001

mnu_m = 0.095
mnu_e = 0.025

omk_m = 0
omk_e = 0

tau_m = 0.054
tau_e = 0.007

ln10As_m = 3.044
ln10As_e = 0.014
As_m = np.exp(ln10As_m)*1e-10
As_e = np.exp(ln10As_e)*1e-10

ns_m = 0.965
ns_e = 0.004

params_m = H0_m,ombh2_m,omch2_m,mnu_m,omk_m,tau_m,As_m,ns_m
params_e = H0_e,ombh2_e,omch2_e,mnu_e,omk_e,tau_e,As_e,ns_e

def params():
    return params_m

def Cls_gen(par, r, lmax):

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=par[0], ombh2=par[1], omch2=par[2], mnu=par[3], omk=par[4],   
                              tau=par[5])
    pars.InitPower.set_params(As=par[6], ns=par[7], r=r)
    pars.WantTensors = True  # Incluir modos tensoriales
    pars.DoLensing = True  # Incluir o no el efecto de lensing
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, raw_cl=True, lmax=lmax, CMB_unit='muK')
    Cls_len = results.get_lens_potential_cls(raw_cl=True, lmax=lmax, CMB_unit='muK')
    
    Cls_cmb = powers['unlensed_total']
    ell = np.arange(Cls_cmb.shape[0])
    
    Cls = {
    'tt': Cls_cmb[:,0],
    'ee': Cls_cmb[:,1],
    'bb': Cls_cmb[:,2],
    'te': Cls_cmb[:,3],
    'pp': Cls_len[:,0],
    'tp': Cls_len[:,1],
    'ep': Cls_len[:,2]}
    
    return Cls, ell

def lensed_cmb_from_spectra(Cls,nside):
    lmax = 3*nside-1
    geom_info = ('healpix', {'nside':nside}) # Geometry parametrized as above, this is the default
    maps = synfast(Cls, lmax=lmax, verbose=0, geometry=geom_info)
    return [maps['T'],maps['QU'][0],maps['QU'][1]]

def freq_foreg_maps(nside,freqs,dust_model,sync_model):
    
    foreg_maps = []
    if sync_model == 0:
        sky_dust = pysm3.Sky(nside=nside, preset_strings=[dust_model])
    else:
        sky_dust = pysm3.Sky(nside=nside, preset_strings=[dust_model,sync_model])
    
    for i in range(len(freqs)):
        
        map_dust_sky = sky_dust.get_emission(freqs[i] * u.GHz)
        map_dust_sky = map_dust_sky.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freqs[i]*u.GHz))
        
        map_dust_q = map_dust_sky.value[1]
        map_dust_u = map_dust_sky.value[2]
        
        foreg_maps.append([map_dust_q,map_dust_u])
        
    return np.array(foreg_maps)

def conv_with_beam(map_TQU, fwhm_grad, lmax):
    conv_maps = hp.smoothing(map_TQU, fwhm=np.radians(fwhm_grad), lmax=lmax, pol=True)
    return conv_maps

def gaussian(x, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))

def noise_raw_map(nside,sen):
    npix = hp.nside2npix(nside)
    Om_pix_amin = (4*np.pi/npix) * (180*60/np.pi) ** 2
    sigma_pix = np.sqrt(sen**2/Om_pix_amin)
    noise = np.random.randn(npix)
    noise *= sigma_pix
    return noise

def coverage_map(nside,center,rad_std,rad_max,rad_num,max_hit):
    rad = np.linspace(0,rad_max,rad_num)
    rad = np.flip(rad)
    vals = gaussian(rad,rad_std)
    vals = vals/np.max(vals)
    hit_dist = max_hit * vals
    hits = np.ones(hp.nside2npix(nside))*hp.UNSEEN
    for i in range(rad_num):
        ipix_disc = hp.query_disc(nside=nside, vec=center, radius=np.radians(rad[i]))
        hits[ipix_disc] = int(hit_dist[i])
    return hits

def add_noise2maps(coverage,noise_raw,cont):
    mask_ones = coverage != hp.UNSEEN
    map_out = np.full_like(coverage, hp.UNSEEN)
    map_out[mask_ones] = cont[mask_ones] + noise_raw[mask_ones] / np.sqrt(coverage[mask_ones])
    mask = np.zeros_like(coverage)
    mask[mask_ones] = 1
    map_out[mask==0] = 0
    return map_out
    
def freq_cont_noise_conv_maps(nside,freqs,beams,coverage,sens,out_freq_i,lmax,Cls,
                              dust_model,sync_model):
    
    cmb_maps = lensed_cmb_from_spectra(Cls,nside)
    foreg_maps = freq_foreg_maps(nside,freqs,dust_model,sync_model)
    
    freq_cont_conv_maps_qu = []
    
    noise_q_o = noise_raw_map(nside,sens[out_freq_i])
    noise_u_o = noise_raw_map(nside,sens[out_freq_i])
    for i in range(len(freqs)):
        
        if i != out_freq_i:
            map_foreg_q = foreg_maps[i][0]
            noise_q = noise_raw_map(nside,sens[i])
            mp_cont_q = map_foreg_q+cmb_maps[1]
            mp_n_cont_q = add_noise2maps(coverage,noise_q,mp_cont_q)
            
            map_foreg_u = foreg_maps[i][1]
            noise_u = noise_raw_map(nside,sens[i])
            mp_cont_u = map_foreg_u+cmb_maps[2]
            mp_n_cont_u = add_noise2maps(coverage,noise_u,mp_cont_u)
            
            maps_cont = np.array([np.zeros_like(cmb_maps[0]),mp_n_cont_q,mp_n_cont_u])
            maps_cont_conv = conv_with_beam(maps_cont, beams[i], lmax)
            maps_cont_conv_qu = np.array([maps_cont_conv[1],maps_cont_conv[2]])
            freq_cont_conv_maps_qu.append(maps_cont_conv_qu)
        
        if i == out_freq_i:
            map_foreg_q = foreg_maps[i][0]
            mp_cont_q = map_foreg_q+cmb_maps[1]
            mp_n_cont_q = add_noise2maps(coverage,noise_q_o,mp_cont_q)
            
            map_foreg_u = foreg_maps[i][1]
            mp_cont_u = map_foreg_u+cmb_maps[2]
            mp_n_cont_u = add_noise2maps(coverage,noise_u_o,mp_cont_u)
            
            maps_cont = np.array([np.zeros_like(cmb_maps[0]),mp_n_cont_q,mp_n_cont_u])
            maps_cont_conv = conv_with_beam(maps_cont, beams[i], lmax)
            maps_cont_conv_qu = np.array([maps_cont_conv[1],maps_cont_conv[2]])
            freq_cont_conv_maps_qu.append(maps_cont_conv_qu)
        
        
    freq_cont_conv_maps_qu = np.array(freq_cont_conv_maps_qu)
    
    mp_n_cmb_q = add_noise2maps(coverage,noise_q_o,cmb_maps[1])
    mp_n_cmb_u = add_noise2maps(coverage,noise_u_o,cmb_maps[2])
    
    maps_cmb = np.array([np.zeros_like(cmb_maps[0]),mp_n_cmb_q,mp_n_cmb_u])
    maps_cmb_conv = conv_with_beam(maps_cmb, beams[out_freq_i], lmax)
    maps_cmb_conv_qu = np.array([maps_cmb_conv[1],maps_cmb_conv[2]])
    
    return freq_cont_conv_maps_qu, maps_cmb_conv_qu

def maps2blocks(maps,nside,freqs):
    
    freq_cont_conv_maps_qu, maps_cmb_conv_qu = maps
    
    maps_cmb_conv_q = maps_cmb_conv_qu[0]
    plane_cmb_conv_q = sphere2piecePlane(maps_cmb_conv_q, nside=nside)
    blocks_cmb_conv_q = piecePlane2blocks(plane_cmb_conv_q, nside=nside)
    block_cmb_conv_q = blocks_cmb_conv_q['block_11']
    
    maps_cmb_conv_u = maps_cmb_conv_qu[1]
    plane_cmb_conv_u = sphere2piecePlane(maps_cmb_conv_u, nside=nside)
    blocks_cmb_conv_u = piecePlane2blocks(plane_cmb_conv_u, nside=nside)
    block_cmb_conv_u = blocks_cmb_conv_u['block_11']
    
    block_cmb_conv_qu = np.array([block_cmb_conv_q,block_cmb_conv_u])
    
    block_cont_conv_qu = []
    for i in range(len(freqs)):
        maps_cont_conv_q = freq_cont_conv_maps_qu[i][0]
        plane_cont_conv_q = sphere2piecePlane(maps_cont_conv_q, nside=nside)
        blocks_cont_conv_q = piecePlane2blocks(plane_cont_conv_q, nside=nside)
        block_cont_conv_q = blocks_cont_conv_q['block_11']
        
        maps_cont_conv_u = freq_cont_conv_maps_qu[i][1]
        plane_cont_conv_u = sphere2piecePlane(maps_cont_conv_u, nside=nside)
        blocks_cont_conv_u = piecePlane2blocks(plane_cont_conv_u, nside=nside)
        block_cont_conv_u = blocks_cont_conv_u['block_11']
        
        block_cont_conv_qu.append([block_cont_conv_q,block_cont_conv_u])
    
    block_cont_conv_qu = np.array(block_cont_conv_qu)
    
    return block_cont_conv_qu, block_cmb_conv_qu

def reduct_tools_from_coverage(coverage):
    nside = hp.npix2nside(len(coverage))
    mask = np.ones(hp.nside2npix(nside))
    mask[coverage==hp.UNSEEN] = 0
    plane_mask = sphere2piecePlane(mask, nside=nside)
    blocks_mask = piecePlane2blocks(plane_mask, nside=nside)
    block_mask = blocks_mask['block_11']
    filas, columnas = np.where(block_mask != 0)
    fila_min, fila_max = filas.min(), filas.max()
    col_min, col_max = columnas.min(), columnas.max()
    max_size = max(fila_max-fila_min,col_max-col_min)
    k=0
    while 2**k <= max_size:
        k+=1
    return fila_min,fila_max,col_min,col_max,2**k

def block_reduction(block,tools,CMB):
    fila_min,fila_max,col_min,col_max,targ_size = tools

    if not CMB:
        smaller_block = block[:,:,fila_min:fila_max,col_min:col_max]
        target_shape = (targ_size, targ_size)
        pad_filas = target_shape[0] - smaller_block.shape[2]
        pad_columnas = target_shape[1] - smaller_block.shape[3]
        pad_width = (
        (0, 0),
        (0, 0),  # no se rellena en dim 0
        (0, pad_filas),  # rellenar solo abajo en dim 1
        (0, pad_columnas)  # rellenar solo a la derecha en dim 2
        )
        #print(block.shape)
        block_filled = np.pad(smaller_block, pad_width, mode='constant', constant_values=0)
    if CMB:
        smaller_block = block[:,fila_min:fila_max,col_min:col_max]
        target_shape = (targ_size, targ_size)
        pad_filas = target_shape[0] - smaller_block.shape[1]
        pad_columnas = target_shape[1] - smaller_block.shape[2]
        pad_width = (
        (0, 0),  # no se rellena en dim 0
        (0, pad_filas),  # rellenar solo abajo en dim 1
        (0, pad_columnas)  # rellenar solo a la derecha en dim 2
        )
        #print(block.shape)
        block_filled = np.pad(smaller_block, pad_width, mode='constant', constant_values=0)
    #print(block_filled.shape)
    return block_filled

def block_upgrade(block,tools,nside):
    fila_min,fila_max,col_min,col_max,targ_size = tools
    h, w = fila_max-fila_min, col_max-col_min
    small_block = block[:, :h, :w]
    orig_block = np.zeros((block.shape[0],nside,nside))
    orig_block[:,fila_min:fila_max,col_min:col_max] = small_block
    return orig_block

def sets_generator(num_train,num_valid,nside,freqs,out_freq_i,beams,sens,coverage,tools,dust_model,
                   sync_model,Cls,lmax,Train,Valid):
    folder_name = dust_model + sync_model
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if Train:
        os.chdir(folder_name)
        for i in tqdm(range(num_train)):
            cont_maps = freq_cont_noise_conv_maps(nside,freqs,beams,coverage,sens,out_freq_i,
                                                  lmax,Cls,dust_model,sync_model)
            freq_cont_conv_block_qu, block_cmb_conv_qu = maps2blocks(cont_maps,nside,freqs)
            
            freq_cont_conv_block_qu = block_reduction(freq_cont_conv_block_qu,tools,CMB=False)
            block_cmb_conv_qu = block_reduction(block_cmb_conv_qu,tools,CMB=True)

            fact = np.max(np.abs(freq_cont_conv_block_qu))
            
            freq_cont_conv_block_qu = freq_cont_conv_block_qu/fact
            block_cmb_conv_qu = block_cmb_conv_qu/fact
            
            np.savez(str(i) +'tr.npz', freq_cont_conv_block_qu=freq_cont_conv_block_qu, 
                     block_cmb_conv_qu=block_cmb_conv_qu, fact=fact)
        os.chdir('..')
            
    if Valid:
        os.chdir(folder_name)
        for i in tqdm(range(num_valid)):
            cont_maps = freq_cont_noise_conv_maps(nside,freqs,beams,coverage,sens,out_freq_i,
                                                  lmax,Cls,dust_model,sync_model)
            freq_cont_conv_block_qu, block_cmb_conv_qu = maps2blocks(cont_maps,nside,freqs)
            
            freq_cont_conv_block_qu = block_reduction(freq_cont_conv_block_qu,tools,CMB=False)
            block_cmb_conv_qu = block_reduction(block_cmb_conv_qu,tools,CMB=True)
            
            fact = np.max(np.abs(freq_cont_conv_block_qu))
            
            freq_cont_conv_block_qu = freq_cont_conv_block_qu/fact
            block_cmb_conv_qu = block_cmb_conv_qu/fact
            
            np.savez(str(i) +'vl.npz', freq_cont_conv_block_qu=freq_cont_conv_block_qu, 
                     block_cmb_conv_qu=block_cmb_conv_qu, fact=fact)
        os.chdir('..')


def frequency_blocks_test(nside,freqs,out_freq_i,beams,sens,coverage,tools,dust_model,
                          sync_model,lmax,Cls):

    cont_maps = freq_cont_noise_conv_maps(nside,freqs,beams,coverage,sens,out_freq_i,
                                          lmax,Cls,dust_model,sync_model)
    
    freq_cont_conv_block_qu, block_cmb_conv_qu = maps2blocks(cont_maps,nside,freqs)
    
    freq_cont_conv_block_qu = block_reduction(freq_cont_conv_block_qu,tools,CMB=False)
    block_cmb_conv_qu = block_reduction(block_cmb_conv_qu,tools,CMB=True)

    fact = np.max(np.abs(freq_cont_conv_block_qu))
    
    freq_cont_conv_block_qu = freq_cont_conv_block_qu/fact
    block_cmb_conv_qu = block_cmb_conv_qu/fact

    return freq_cont_conv_block_qu, block_cmb_conv_qu, fact