class Force_Constants_BCC:
    def __init__(self, coordinates=[], fc_matrices=[]):
        self.coordinates = coordinates
        self.fc_matrices = fc_matrices

    def add(self, displ_coords, fc_matrix_prototype):
        self.coordinates.append(displ_coords)
        self.fc_matrices.append(fc_matrix_prototype)

    def swap_elements_matrix(self, matrix, swap):
        i, j = swap
        matrix[[i,j]] = matrix[[j,i]]
        matrix.T[[i,j]] = matrix.T[[j,i]]

    def chg_sign_matrix(self, matrix, i):
        matrix[[i]]=-matrix[[i]]
        matrix.T[[i]]=-matrix.T[[i]]

    def swap_elements_list(self, list, swap):
        i, j = swap
        l = list.copy()
        l[i],l[j] = l[j],l[i]
        return l

    def find_swaps(self, candidate, target):
        c = list(map(abs, candidate))
        t = list(map(abs, target))
        # check null swap
        if c == t:
            return (0,0)
        # check 0 <-> 1 swap
        if self.swap_elements_list(c,(0,1)) == t:
            return (0,1)
        # check 0 <-> 2 swap
        if self.swap_elements_list(c,(0,2)) == t:
            return (0,2)
        # check 1 <-> 2 swap
        if self.swap_elements_list(c,(1,2)) == t:
            return (1,2)
        return False


    def gen_fc_matrix(self, input_coords):
        #check against every stored fc_matrix for a match.
        for coordinate, fc_matrix in zip(self.coordinates, self.fc_matrices):
            swap = self.find_swaps(input_coords, coordinate)
            if swap:
                break
        else:
            return None
        # Copy fc_matrix
        out_matrix = fc_matrix.copy()
        # Find changes in sign and apply
        swapped_in = self.swap_elements_list(input_coords, swap)
        for i in range(3):
            if (swapped_in[i]*coordinate[i] < 0):
                self.chg_sign_matrix(out_matrix, i)
        # Apply swap
        self.swap_elements_matrix(out_matrix, swap)
        return out_matrix


import numpy as np
f = Force_Constants_BCC()

c1=[0,0,1]
fc1=np.array([[1,2,3],[2,4,5],[3,5,9]])
f.add(c1,fc1)
r = f.gen_fc_matrix([0,0,1])
print(r)




