import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.io import write
from ase.io.vasp import write_vasp, read_vasp
from copy import copy
from scipy.spatial.distance import cdist, euclidean
import pandas as pd
import pymatgen as mg
import fc_matcher as fcm

'''
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
'''

# Create a pymatgen lattice, lengths are in Angstroms.
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
# NOTE: dyne/cm -> eV/A**2, check zeros
c1xx =  7798.4
c1xy =  8341.7
c2xx =  4960.5
c2yy = -2170.2
c3xx =   838.6
c3zz = -1410.0
c3xy =   866.9
c4xx =    45.5
c4yy =   204.4
c4yz =  -895.4
c4xy =  -134.9
c5xx =   172.1
c5xy =   703.0


# Load force constants as matrices into fc matcher.
fc = fcm.Force_Constants_BCC()

# Values should be sum of others (Bruesch).
fc.add([0,0,0],np.array([[0   ,0   ,0   ],
                         [0   ,0   ,0   ],
                         [0   ,0   ,0   ]]))

fc.add([1,1,1],np.array([[c1xx,c1xy,c1xy],
                         [c1xy,c1xx,c1xy],
                         [c1xy,c1xy,c1xx]]))

fc.add([2,0,0],np.array([[c2xx,0   ,0   ],
                         [0   ,c2yy,0   ],
                         [0   ,0   ,c2yy]]))

fc.add([2,2,0],np.array([[c3xx,c3xy,0   ],
                         [c3xy,c3xx,0   ],
                         [0   ,0   ,c3zz]]))
fc.add([3,1,1],np.array([[c4xx,c4xy,c4xy],
                         [c4xy,c4yy,c4yz],
                         [c4xy,c4yz,c4yy]]))
fc.add([2,2,2],np.array([[c5xx,c5xy,c5xy],
                         [c5xy,c5xx,c5xy],
                         [c5xy,c5xy,c5xx]]))

# Hacky MIC, add [3,3,1] and [3,3,3]. Both maapped to [1,1,1]

#fc.add([3,3,1], fc.gen_fc_matrix([1,1,1]))
#fc.add([3,3,3], fc.gen_fc_matrix([1,1,1]))

#fc.add([3,1,1], fc.gen_fc_matrix([1,1,1]))

# Generate FORCE_CONSTANTS file.

N = len(struct.sites)
fc_filename = 'FORCE_CONSTANTS'
fc_file = open(fc_filename,"w")
# Write total number of atoms to file.
fc_file.write(str(N)+' '+str(N)+'\n')

# Itereate over pairs of of atoms (sites) in supercell.
for index1, site1 in enumerate(struct.sites):
    for index2, site2 in enumerate(struct.sites):
        # Write the index of the two current atoms to file.
        fc_file.write(str(index1)+' '+str(index2)+'\n')
        # Displ_coord is calcualted as a numpy array.
        displ_coord = (site2.frac_coords-site1.frac_coords)*4
        # Convert individual elements to list of ints.
        displ_coord = list(map(int, map(round, displ_coord)))
        # Delete the following line.
        #fc_file.write(str(displ_coord)+'\n')
        '''
        print('new iter:')
        print(type(displ_coord))
        print(displ_coord)
        print()
        print()
        '''
        fc_matrix = fc.gen_fc_matrix(list(displ_coord))
        # If no fc_matrix matches, use zeros.
        if fc_matrix is None:
            fc_matrix = fc.gen_fc_matrix([0,0,0])
        # Form string from matrix, remove brackets, and write to file.
        matrix_string = str(fc_matrix).replace('[','').replace(']','')+'\n'
        fc_file.write(matrix_string)
fc_file.close()


# Create 4-D array of zeros to fill in with a 3x3 Force Constant matrix
# corresponding to each pair of atoms, i,j. 
#
# E.g.   (2,3,x,y) will be the value of the x-component of the force acting
#        on the 2nd atom when the 3rd atom moves in the y direction.
'''
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

fc_filename = 'FORCE_CONSTANTS_TEST'
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
'''
