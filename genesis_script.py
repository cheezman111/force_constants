import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.io import write
from ase.io.vasp import write_vasp, read_vasp
from copy import copy
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
import pymatgen as mg


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
        nns = np.array(([0,0,0],[1,1,1],[2,0,0],[2,2,0],[3,1,1],[2,2,2])) \
              * 0.5 * latt_param
        nn_dists = np.array([euclidean(nn, np.zeros(3)) for nn in nns])

    nn = 0 # Error
    for c, nn_dist in enumerate(nn_dists):
        if verbose: print(nn_dist - tol, dx, nn_dist + tol)
        if nn_dist - tol < dx and dx < nn_dist + tol:
            nn = c

    return nn



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


# Create a pymatgen lattice, length units are in Angstroms.
lat = mg.Lattice.cubic(3.5175)
# Produce a BCC structure with Zr atoms.
struct = mg.Structure(lat, ['Zr','Zr'], [[0,0,0],[.5,.5,.5]] )
# Expand to supercell with 16 total atoms.
struct.make_supercell([2,2,2])
# Create poscar object and write to file.
pc = mg.io.vasp.inputs.Poscar(struct)
pc.write_file('POSCAR')


# Hardcode force constant matrices for for 0th-5th nearest neighbors using 
# experimental values from Heiming-1991.
fc_Zr_1210 = np.array(([[0, 0, 0],[0, 0, 0],[0, 0, 0]],
                       [[7798, 0, 0],[0, 8341, 0],[0, 0, 8341]],
                       [[4960, 0, 0],[0, -2170, 0],[0, 0, -2170]],
                       [[838, 866, 0],[866, 1410, 0],[0, 0, 1410]],
                       [[45, -134, -134],[-134, 204, -995],[-134, -995, 204]],
                       [[172, 703, 703],[703, 172, 703],[703, 703, 172]]), dtype=float) #dyne/cm -> eV/A**2, check zeros


# Create 4-D array of zeros to fill in with a 3x3 Force Constant matrix
# corresponding to each pair of atoms, i,j. 
#
# E.g.   (2,3,x,y) will be the value of the x-component of the force acting
#        on the 2nd atom when the 3rd atom moves in the y direction.
atom_pairs = []
nearest_neighbors = []
fc_arr = np.zeros((len(coords), len(coords), 3, 3), dtype=float)
print(len(coords),len(coords))
for i, coord1 in enumerate(coords):
    for j, coord2 in enumerate(coords):

        dx = euclidean(coord1,coord2)
        nn = calc_nn(dx=dx, latt_param=d/2, latt_type='bcc')
        #------
        atom_pairs.append((i,j))
        nearest_neighbors.append(nn)
        #------
        if -1 == nn:
            print('Problem:', i, j, dx)

        fc_arr[i][j][:] = fc_Zr_1210[nn]

fc_filename = 'FORCE_CONSTANTS'
fc_file = open(fc_filename,"w")
fc_file.write(str(len(coords)) + ' ' + str(len(coords)) + '\n')

for i, coord1 in enumerate(coords):

    translation_fcs = np.zeros((3, 3), dtype=float)
    for j, coord2 in enumerate(coords):
        translation_fcs = translation_fcs + fc_arr[i][j][:]

        fc_file.write(str(i+1) + ' ' + str(j+1) + '\n')
        fc_file.write(' '.join(map(str,fc_arr[i][j][0])) + '\n')
        fc_file.write(' '.join(map(str,fc_arr[i][j][1])) + '\n')
        fc_file.write(' '.join(map(str,fc_arr[i][j][2])) + '\n')

fc_file.close()
print('Done.')

