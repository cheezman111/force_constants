class Force_Constants_BCC:
    def __init__(self, coordinates=[], fc_matrices=[]):
        self.coordinates = coordinates
        self.fc_matrices = fc_matrices

    def add(self, displ_coords, fc_matrix_prototype):
        self.coordinates.append(displ_coords)
        self.fc_matrices.append(fc_matrix_prototype)

    def swap_indices(self, matrix, i, j):
        matrix[[i,j]] = matrix[[j,i]]
        matrix.T[[i,j]] = matrix.T[[j,i]]

    def chg_sign_index(self, matrix, i):
        matrix[[i]]]=-matrix[[i]]
        matrix.T[[i]]]=-matrix.T[[i]]

    def check_match(self, displ_coords1, displ_coords2):
        indices1 = list(zip(displ_coords1, range(3)))
        indices2 = list(zip(displ_coords2, range(3)))
        sort_f = lambda x : (abs(x[0]),x[1])

        indices1.sort(key=sort_f)
        indices2.sort(key=sort_f)

        d1, i1 = zip(*indices1)
        d2, i2 = zip(*indices2)

        if d1 != d2:
            return None

        chg_sign = [(d1[i]*d2[i] < 0) for i in range(3)]
        swaps = list(zip(i1,i2,chg_sign))
        swaps.sort()
        print(d1)
        print(d2)
        print(i1)
        print(i2)

        return swaps

    def gen_fc_matrix(self, input_coords):

        #check against every store fc_matrix for a match.

        for coordinate, fc_matrix in zip(self.coordinates, self.fc_matrices):
            match = self.check_match(input_coords, coordinate)
            if match is not None:
                break
        else:
            return None

        i1, i2, chg_sign = zip(*match)
        i1=list(i1)
        i2=list(i2)
        fc_matrix[i1] = fc_matrix[i2]

        return fc_matrix


import numpy as np
f = Force_Constants_BCC()

c1=[0,0,1]
fc1=np.array([[1,2,3],[2,4,5],[3,5,9]])
f.add(c1,fc1)
r = f.gen_fc_matrix([0,0,1])
print(r)




