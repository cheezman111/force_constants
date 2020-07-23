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
import h5py

supercell_size = 4

# Create a pymatgen lattice, lengths are in Angstroms.
lat = mg.Lattice.cubic(3.5175)
# Produce a BCC structure with Zr atoms.
struct = mg.Structure(lat, ['Zr','Zr'], [[0,0,0],[.5,.5,.5]] )
# Expand to supercell with 16 total atoms.
struct.make_supercell([supercell_size]*3)
# Create poscar object and write to file.
pc = mg.io.vasp.inputs.Poscar(struct)
pc.write_file('POSCAR')


# Hardcode force constant matrices for for 0th-5th nearest neighbors using
# experimental values from Heiming-1991.
# NOTE: dyne/cm -> eV/A**2, check zeros
unit_conv = .000062415
c0xx =  75848.0*unit_conv
c1xx =   7798.4*unit_conv
c1xy =   8341.7*unit_conv
c2xx =   4960.5*unit_conv
c2yy =  -2170.2*unit_conv
c3xx =    838.6*unit_conv
c3zz =  -1410.0*unit_conv
c3xy =    866.9*unit_conv
c4xx =     45.5*unit_conv
c4yy =    204.4*unit_conv
c4yz =   -995.4*unit_conv
c4xy =   -134.9*unit_conv
c5xx =    172.1*unit_conv
c5xy =    703.0*unit_conv

# Load force constants as matrices into fc matcher.
fc = fcm.Force_Constants_BCC()

# Values should be sum of others (Bruesch).
fc.add([0,0,0],np.array([[c0xx,0.  ,0.  ],
                         [0.  ,c0xx,0.  ],
                         [0.  ,0.  ,c0xx]]))

fc.add([1,1,1],np.array([[c1xx,c1xy,c1xy],
                         [c1xy,c1xx,c1xy],
                         [c1xy,c1xy,c1xx]]))

fc.add([2,0,0],np.array([[c2xx,0.  ,0.  ],
                         [0.  ,c2yy,0.  ],
                         [0.  ,0.  ,c2yy]]))

fc.add([2,2,0],np.array([[c3xx,c3xy,0.  ],
                         [c3xy,c3xx,0.  ],
                         [0.  ,0.  ,c3zz]]))

#'''
fc.add([3,1,1],np.array([[c4xx,c4xy,c4xy],
                         [c4xy,c4yy,c4yz],
                         [c4xy,c4yz,c4yy]]))
#'''

fc.add([2,2,2],np.array([[c5xx,c5xy,c5xy],
                         [c5xy,c5xx,c5xy],
                         [c5xy,c5xy,c5xx]]))

# Hacky MIC, add [3,3,1] and [3,3,3]. Both maapped to [1,1,1]

fc.add([3,3,1], fc.gen_fc_matrix([1,1,1]))
fc.add([3,3,3], fc.gen_fc_matrix([1,1,1]))

#fc.add([3,1,1], fc.gen_fc_matrix([1,1,1]))


# Create the force constants tensor. 
fc_tensor = []
for site1 in struct.sites:
    fc_tensor.append([])
    for site2 in struct.sites:
        # Displ_coord is calcualted as a numpy array.
        displ_coord = (site2.frac_coords-site1.frac_coords)*2*supercell_size
        # Convert individual elements to list of ints.
        displ_coord = list(map(int, map(round, displ_coord)))
        fc_matrix = fc.gen_fc_matrix(list(displ_coord))
        # If no fc_matrix matches, use zeros.
        if fc_matrix is None:
            fc_matrix = np.zeros((3,3))
        fc_tensor[-1].append(fc_matrix)
fc_tensor = np.array(fc_tensor)

self_forces = np.sum(fc_tensor, axis=1)


'''
for index in range(fc_tensor.shape[0]):
    fc_tensor[index, index] = -self_forces[index]
'''


# Write force constants tensor to hdf5 file
hf = h5py.File("force_constants.hdf5", "w")
fc_dataset = hf.create_dataset("force_constants", data=fc_tensor)
hf.close()


'''
# OBSOLETE IF USING HDF5 FILE
# Generate FORCE_CONSTANTS file.

N = len(struct.sites)
fc_filename = 'FORCE_CONSTANTS'
fc_file = open(fc_filename,"w")
# Write total number of atoms to file.
fc_file.write('{:3} {:3}'.format(N,N) + '\n')
# Set numpy print options to print to 15 digits precision.
np.set_printoptions(precision=15, sign=' ',floatmode='fixed')

# Iterate over pairs of of atoms (sites) in supercell.
for index1, site1 in enumerate(struct.sites):
    for index2, site2 in enumerate(struct.sites):
        # Write the index of the two current atoms to file.
        fc_file.write('{:3} {:3}'.format(index1,index2) + '\n')
        # Displ_coord is calcualted as a numpy array.
        displ_coord = (site2.frac_coords-site1.frac_coords)*4
        # Convert individual elements to list of ints.
        displ_coord = list(map(int, map(round, displ_coord)))
        # Delete the following line.
        #fc_file.write(str(displ_coord)+'\n')
        fc_matrix = fc.gen_fc_matrix(list(displ_coord))
        # If no fc_matrix matches, use zeros.
        if fc_matrix is None:
            fc_matrix = fc.gen_fc_matrix([0,0,0])
        # Form string from matrix, remove brackets, and write to file.
        matrix_string = str(fc_matrix).replace('[',' ').replace(']',' ')+'\n'
        fc_file.write(matrix_string)
fc_file.close()
'''
