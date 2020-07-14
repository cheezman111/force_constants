# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.io import write
from ase.io.vasp import write_vasp, read_vasp
from copy import copy
from scipy.spatial.distance import cdist, euclidean
import pandas as pd


d = 7.035
seed = [float(x)/2 for x in range(0, 2)] # 16-atom supercell
coords = []
for i in seed:
    for j in seed:
        for k in seed:
            coords.append((i, j, k))
            coords.append((i+.25, j+.25, k+.25))

coords = np.array(coords)
coords = coords*d

atoms = Atoms(['Zr' for x in range(len(coords))], positions=coords, cell=[d, d, d], pbc=True)


filename = 'POSCAR'

write_vasp(filename, atoms, direct=False)
print('poscar.')


def rectify_x(arr, cell_side):
    rectified = np.zeros(3)
    for i, elem in enumerate(arr):
        elem = elem - cell_side if elem >= cell_side * 0.5 else elem
        rectified[i] = elem
    return rectified

def rectify_dx(arr1, arr2, cell_side):
    rectified = np.zeros(3)
    for i, elem in enumerate(arr1):
        dx = arr2[i] - arr1[i]
        dx = dx - cell_side if dx > cell_side * 0.5 else dx
        dx = dx + cell_side if dx <= -cell_side * 0.5 else dx
        rectified[i] = dx
    return euclidean(rectified, np.zeros(3))

def calc_nn(dx, latt_param, latt_type='bcc', tol=1e-1, verbose=False):
    """
    Current max is 5nn and only for bcc
    """

    if 'bcc' == latt_type:
        nns = np.array(([0,0,0],[1,1,1],[2,0,0],[2,2,0],[3,1,1],[2,2,2])) * 0.5 * latt_param
        nn_dists = np.array([euclidean(nn, np.zeros(3)) for nn in nns])

    nn = -1 # Error
    for c, nn_dist in enumerate(nn_dists):
        if verbose: print(nn_dist - tol, dx, nn_dist + tol)
        if nn_dist - tol < dx and dx < nn_dist + tol:
            nn = c

    return nn


fc_Zr_1210 = np.array(([[0, 0, 0],[0, 0, 0],[0, 0, 0]],
                       [[7798, 0, 0],[0, 8341, 0],[0, 0, 8341]],
                       [[4960, 0, 0],[0, -2170, 0],[0, 0, -2170]],
                       [[838, 866, 0],[866, 1410, 0],[0, 0, 1410]],
                       [[45, -134, -134],[-134, 204, -995],[-134, -995, 204]],
                       [[172, 703, 703],[703, 172, 703],[703, 703, 172]]), dtype=float) #dyne/cm -> eV/A**2, check zeros

fc_arr = np.zeros((len(coords), len(coords), 3, 3), dtype=float)
print(len(coords),len(coords))
for i, coord1 in enumerate(coords):
    for j, coord2 in enumerate(coords):

        coord1 = rectify_x(coord1, cell_side=d)
        coord2 = rectify_x(coord2, cell_side=d)
        dx = rectify_dx(coord1, coord2, cell_side=d)
        nn = calc_nn(dx=dx, latt_param=d/2, latt_type='bcc')

        if -1 == nn:
            print('Problem:', i, j, dx)

        fc_arr[i][j][:] = fc_Zr_1210[nn]

        # print(i,j)
        # print(' '.join(map(str,fc_Zr_1210[nn][0])))
        # print(' '.join(map(str,fc_Zr_1210[nn][1])))
        # print(' '.join(map(str,fc_Zr_1210[nn][2]))) # should be written with the correct symmetry

fc_filename = 'FORCE_CONSTANTS'
fc_file =  open(fc_filename,"w")
fc_file.write(str(len(coords)) + ' ' + str(len(coords)) + '\n')

for i, coord1 in enumerate(coords):

    # if i == j: # fix
    #     continue

    translation_fcs = np.zeros((3, 3), dtype=float)
    for j, coord2 in enumerate(coords):
        translation_fcs = translation_fcs + fc_arr[i][j][:]

    # print(translation_fcs)
    # print(' ')

        fc_file.write(str(i+1) + ' ' + str(j+1) + '\n')
        fc_file.write(' '.join(map(str,fc_arr[i][j][0])) + '\n')
        fc_file.write(' '.join(map(str,fc_arr[i][j][1])) + '\n')
        fc_file.write(' '.join(map(str,fc_arr[i][j][2])) + '\n')

fc_file.close()
print('Done.')

