#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:20:58 2025

@author: lgomez
"""


import numpy as np
import healpy as hp


class Cut(object):
    '''
    cut a Healpix map to 12 parts, or to 12*subblocks_nums parts
    '''

    def __init__(self, maps_in, subblocks_nums=1, nest=False):
        '''
        :param maps_in: the input healpix maps (one map with the shape of (12*nside**2,) or multiple maps with the shape of (N, 12*nside**2))
        :param nest: bool, if False map_in is assumed in RING scheme, otherwise map_in is NESTED
        :param subblocks_nums: int, the number after dividing the square nside*nside into small squares,
                               subblocks_nums=1^2, 2^2, 4^2, 8^2, 16^2, ..., default 1
        '''
        self.maps_in = maps_in
        self.nest = nest
        self.subblocks_nums = subblocks_nums

    @property
    def _multi_map(self):
        if len(self.maps_in.shape) == 1:
            return False
        elif len(self.maps_in.shape) == 2:
            self.multi_map_n = self.maps_in.shape[0]
            return True

    @property
    def nside(self):
        if self._multi_map:
            nsd = int(np.sqrt(self.maps_in[0].shape[0] / self.subblocks_nums / 12))
        else:
            nsd = int(np.sqrt(self.maps_in.shape[0] / self.subblocks_nums / 12))
        return nsd

    def _expand_array(self, original_array):
        '''
        to be used in nestedArray2nestedMap, expand the given small array into a large array

        :param original_array: with the shape of (2**n, 2**n), where n=1, 2, 4, 6, 8, 10, ...
        '''
        add_value = original_array.shape[0] ** 2
        array_0 = original_array
        array_1 = array_0 + add_value
        array_2 = array_0 + add_value * 2
        array_3 = array_0 + add_value * 3
        array_3_1 = np.c_[array_3, array_1]
        array_2_0 = np.c_[array_2, array_0]
        array = np.r_[array_3_1, array_2_0]
        return array

    def _ordinal_array(self):
        '''
        obtain an array containing the ordinal number, the shape is (nside, nside)
        '''
        circle_num = (int(np.log2(self.nside ** 2)) - 2) // 2  # use //
        ordinal_array = np.array([[3., 1.], [2., 0.]])
        for i in range(circle_num):
            ordinal_array = self._expand_array(ordinal_array)
        return ordinal_array, circle_num

    def nestedArray2nestedMap(self, map_cut):
        '''
        reorder the cut map into NESTED ordering to show the same style using
        plt.imshow() as that using Healpix

        :param map_cut: the cut map, the shape of map_cut is (nside**2,)

        return the reorded data, the shape is (nside, nside)
        '''
        array_fill, circle_num = self._ordinal_array()
        # for i in range(2 ** (circle_num + 1)):
        #     for j in range(2 ** (circle_num + 1)):
        #         array_fill[i][j] = map_cut[int(array_fill[i][j])]
        array_fill = map_cut[array_fill.reshape(1, -1).astype(int)].reshape(2 ** (circle_num + 1),
                                                                            2 ** (circle_num + 1))

        # array_fill should be transposed to keep the figure looking like that in HEALPix

        return array_fill

    def nestedMap2nestedArray(self, map_block):
        '''
        Restore the cut map(1/12 of full sky map) into an array which is in NESTED ordering

        need transpose if the map is transposed in nestedArray2nestedMap function
        '''
        map_cut = np.ones(self.nside ** 2)
        array_fill, circle_num = self._ordinal_array()
        for i in range(2 ** (circle_num + 1)):
            for j in range(2 ** (circle_num + 1)):
                map_cut[int(array_fill[i][j])] = map_block[i][j]
        return map_cut

    def _block(self, Map, block_n):
        '''
        return one block of one of the original maps
        '''
        if not self.nest:
            # reorder the map from RING ordering to NESTED
            map_NEST = hp.reorder(Map, r2n=True)

        map_part = map_NEST[block_n * self.nside ** 2: (block_n + 1) * self.nside ** 2]
        map_part = self.nestedArray2nestedMap(map_part)
        return map_part

    def block(self, block_n):
        if self._multi_map:
            map_part = []
            for i in range(self.multi_map_n):
                map_part.append(self._block(self.maps_in[i], block_n))
        else:
            map_part = self._block(self.maps_in, block_n)
        return map_part

    def _block_all(self, Map):
        '''
        return all blocks (12 blocks) of one of the original maps
        '''
        map_parts = []
        for blk in range(12 * self.subblocks_nums):
            map_parts.append(self._block(Map, blk))
        return map_parts

    def block_all(self):
        if self._multi_map:
            map_parts = []
            for i in range(self.multi_map_n):
                map_parts.append(self._block_all(self.maps_in[i]))
        else:
            map_parts = self._block_all(self.maps_in)
        return map_parts


class Block2Full(Cut):
    '''
    stitch a cut map (1/12 of full sky map) to a full sky map with other parts is zeros
    '''

    def __init__(self, maps_block, block_n, subblocks_nums=1, base_map=None):
        '''
        :param maps_block: the cut map in NESTED ording (one map in 2D array with the shape of (nside, nside)
        or multiple maps in 3D array with the shape of (N, nside, nside) or multiple maps in a list with each element has the shape of (nside,nside))
        :param block_n: int, the number of cut map, 0, 1, 2, ..., 11
        :param subblocks_nums: int, the number after dividing the square nside*nside into small squares,
                               subblocks_nums=1^2, 2^2, 4^2, 8^2, 16^2, ..., default 1
        '''
        self.maps_block = maps_block
        self.block_n = block_n
        self.base_map = base_map
        self.subblocks_nums = subblocks_nums

    @property
    def _multi_map(self):
        # list -> array
        if isinstance(self.maps_block, list):
            self.maps_block = np.array(self.maps_block)

        if len(self.maps_block.shape) == 2:
            return False
        elif len(self.maps_block.shape) == 3:
            self.multi_map_n = self.maps_block.shape[0]
            return True

    @property
    def nside(self):
        if self._multi_map:
            nsd = self.maps_block.shape[1]
        else:
            nsd = self.maps_block.shape[0]
        return nsd

    def _full(self, map_block):
        '''
        return a full sphere map
        '''
        map_block_array = self.nestedMap2nestedArray(map_block)
        if self.base_map is None:
            map_NEST = np.zeros(12 * self.subblocks_nums * self.nside ** 2)
        else:
            map_NEST = hp.reorder(self.base_map, r2n=True)
        map_NEST[self.block_n * self.nside ** 2: (self.block_n + 1) * self.nside ** 2] = map_block_array
        map_RING = hp.reorder(map_NEST, n2r=True)
        return map_RING

    def full(self):
        if self._multi_map:
            map_full = []
            for i in range(self.multi_map_n):
                map_full.append(self._full(self.maps_block[i]))
        else:
            map_full = self._full(self.maps_block)
        return map_full


def sphere2piecePlane(sphere_map, nside=256):
    '''
    cut full map to 12 blocks, then piecing them together into a plane
    this is only for the case of subblocks_nums=1
    '''
    blocks = Cut(sphere_map).block_all()

    piece_map = np.zeros((nside * 4, nside * 3))
    # part 1
    piece_map[nside * 3:, :nside] = blocks[1]  # block 1
    piece_map[nside * 3:, nside:nside * 2] = blocks[5]
    piece_map[nside * 3:, nside * 2:] = blocks[8]
    # part 2
    piece_map[nside * 2:nside * 3, :nside] = blocks[0]
    piece_map[nside * 2:nside * 3, nside:nside * 2] = blocks[4]
    piece_map[nside * 2:nside * 3, nside * 2:] = blocks[11]
    # part 3
    piece_map[nside:nside * 2, :nside] = blocks[3]
    piece_map[nside:nside * 2, nside:nside * 2] = blocks[7]
    piece_map[nside:nside * 2, nside * 2:] = blocks[10]
    # part 4
    piece_map[:nside, :nside] = blocks[2]
    piece_map[:nside, nside:nside * 2] = blocks[6]
    piece_map[:nside, nside * 2:] = blocks[9]
    return piece_map

def piecePlane2blocks(piece_map, nside=256):
    '''
    :param Map: plane map whose shape is (nside*4, nside*3)
    this is only for the case of subblocks_nums=1
    '''
    blocks = {}
    # part 1
    blocks['block_1'] = piece_map[nside * 3:, :nside]  # block 1
    blocks['block_5'] = piece_map[nside * 3:, nside:nside * 2]
    blocks['block_8'] = piece_map[nside * 3:, nside * 2:]
    # part 2
    blocks['block_0'] = piece_map[nside * 2:nside * 3, :nside]
    blocks['block_4'] = piece_map[nside * 2:nside * 3, nside:nside * 2]
    blocks['block_11'] = piece_map[nside * 2:nside * 3, nside * 2:]
    # part 3
    blocks['block_3'] = piece_map[nside:nside * 2, :nside]
    blocks['block_7'] = piece_map[nside:nside * 2, nside:nside * 2]
    blocks['block_10'] = piece_map[nside:nside * 2, nside * 2:]
    # part 4
    blocks['block_2'] = piece_map[:nside, :nside]
    blocks['block_6'] = piece_map[:nside, nside:nside * 2]
    blocks['block_9'] = piece_map[:nside, nside * 2:]
    return blocks
 
def blockPlane2sphere_mult(block_maps, nside=512, block_n='block_11'):
    if len(block_maps.shape) == 2:
        multi_map = False
    elif len(block_maps.shape) == 3:  # shape: [freq, pix, pix]
        multi_map = True
        multi_map_n = block_maps.shape[0]
    if multi_map:
        base_map = np.zeros((multi_map_n, 12 * nside ** 2))
        for i in range(multi_map_n):
            full_map_i = Block2Full(block_maps[i].swapaxes(-1,-2), block_n=11, base_map=None).full() # Note! .T
            base_map[i, :] = full_map_i
        return base_map
    else:
        return Block2Full(block_maps, block_n=11, base_map=None).full() # Note! .T