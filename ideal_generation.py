import pymatgen as mg
import math
import copy

k_master = .5

def find_fc_matrix(n1, n2, positions, differential):
    matrix = []
    for alpha in range(3):
        matrix.append([])
        for beta in range(3):
            fc = find_force_constant(n1,n2,alpha,beta,positions,differential)
            matrix[-1].append(fc)
    return matrix


def find_force_constant(n1, n2, alpha, beta, positions, differential):
    positions1 = positions
    positions2 = copy.deepcopy(positions)
    positions2[n2][beta] += differential

    translation1 = list(map(lambda x,y:x-y, positions1[n2], positions1[n1]))
    translation2 = list(map(lambda x,y:x-y, positions2[n2], positions2[n1]))

    # Calcualte forces. These are the full magnitudes, not components.
    k = k_master/math.log(length(translation1))
    force1 = calculate_force(translation1,k)
    force2 = calculate_force(translation2,k)

    # Calculate force components in alpha direction
    force1_alpha = force1*translation1[alpha]/length(translation1)
    force2_alpha = force2*translation2[alpha]/length(translation2)

    return (force2_alpha-force1_alpha)/differential


def calculate_force(r,k):
    return -1*k*length(r)

def length(r):
    return math.sqrt(sum(map(lambda x:x**2,r)))


n = 2
positions = []
for x in range(n):
    for y in range(n):
        for z in range(n):
            positions.append([x,y,z])
            positions.append([x+.5,y+.5,z+.5])





