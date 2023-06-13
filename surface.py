import numpy as np
import os
import math
import pandas as pd

PI = math.pi 

def zip_arrays(arr_even, arr_odd):
    zipped_array = [val for pair in zip(arr_even, arr_odd) for val in pair]
    return zipped_array

def is_in_wedge(P,alpha,beta):
    """
    This checks whether a point P lies in the sector of a circle spanned by the angles alpha and beta.
    Returns:
        True if P lies to the left of alpha
        None if P is contained in the sector of the circle
        False if P lies to the right of beta
    """
    left = None
    phi = math.atan2(P[1], P[0])
    if phi <= alpha:
        left = True
    elif phi > beta:
        left = False
    return left


def find_wedge_recursion(P, n, current_state, shift=0):
    """
    This is a divide and concer algorithm which determines the number of the wedge of a (2^n-1)-gon
    It runs in log(n).
    
    Returns:
        int: the wedge number
    """
    alpha = (2.*shift - 1.) * PI / (2**n - 1)
    beta = (2.*shift + 1.) * PI / (2**n - 1)
    current_state -= 1
    left = is_in_wedge(P,alpha,beta)
    if left is None:
        return 2**(n-1) + shift
    elif left is True:
        shift -= current_state 
        if current_state == 1:
            return 2**(n-1) + shift
        else:
            return find_wedge_recursion(P,n,current_state,shift)
    else:
        shift += current_state
        if current_state == 1:
            return 2**(n-1) + shift
        else:
            return find_wedge_recursion(P,n,current_state,shift)


def find_wedge(P, n):
    """
    This starts the recursion. 
    """
    return find_wedge_recursion(P,n,n)

   

def test_find_wedge():
    phi = -math.pi/4.
    x = math.cos(phi)
    y = math.sin(phi)
    P = np.arange(2.)
    P[0] = x 
    P[1] = y
    result = find_wedge(P,3)
    assert result == 3, f"expected wedge is 3, got: {result}"
    phi = math.pi/4.
    x = math.cos(phi)
    y = math.sin(phi)
    P = np.arange(2.)
    P[0] = x 
    P[1] = y
    result = find_wedge(P,3)
    assert result == 5, f"expected wedge is 5, got: {result}"

test_find_wedge()

def set_wedge_array(points, n):
    vfunc = np.vectorize(find_wedge, signature="(n),() -> ()")
    #vfunc = np.vectorize(find_wedge)
    result = vfunc(points, n)
    return result

def test_wedge_array():
    phi = -math.pi/4.
    x = math.cos(phi)
    y = math.sin(phi)
    P1 = np.arange(2.)
    P1[0] = x 
    P1[1] = y
    P2 = 2.*P1
    phi = math.pi/4.
    x = math.cos(phi)
    y = math.sin(phi)
    P3 = np.arange(2.)
    P3[0] = x 
    P3[1] = y
    points = []
    points = list((P1, P2, P3))
    points = np.array(points)
    #print(points[0])
    wedges = set_wedge_array(points,3)
    expected_wedges = np.array([3,3,5])
    assert np.allclose(wedges, expected_wedges), "test_wedge_array(): do not match the expected values."
    

test_wedge_array()

def split_into_pos_neg_z_values(P):
        third_component = P[:,2]
        pos_idx = np.where(third_component > 0)
        neg_idx = np.where(third_component <= 0)
        return pos_idx, neg_idx


def split_into_pos_neg_pts(P):
        pos_idx, neg_idx = split_into_pos_neg_z_values()
        pos_pts = P[pos_idx]
        neg_pts = P[neg_idx]
        return pos_pts, neg_pts




class Surface:
    def __init__(self, surface):
        # find path to the directory
        dir_path = os.path.dirname(os.path.abspath(__file__))

        if surface == 'Minor Ellipsoid':
            path = os.path.join(dir_path, 'data/minor_ellipsoid/eigenvalues.csv')
            eigenvalues = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/minor_ellipsoid/eigenvectors.csv')
            eigenvectors = pd.read_csv(path,header=None)
            # load surface coordinate
            path = os.path.join(dir_path, 'data/minor_ellipsoid/Px.csv')
            Px = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/minor_ellipsoid/Py.csv')
            Py = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/minor_ellipsoid/Pz.csv')
            Pz = pd.read_csv(path,header=None)
            

        elif surface == 'Major Ellipsoid':
            path = os.path.join(dir_path, 'data/major_ellipsoid/eigenvalues.csv')
            eigenvalues = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/major_ellipsoid/eigenvectors.csv')
            eigenvectors = pd.read_csv(path,header=None)
            # load surface coordinate
            path = os.path.join(dir_path, 'data/major_ellipsoid/Px.csv')
            Px = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/major_ellipsoid/Py.csv')
            Py = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/major_ellipsoid/Pz.csv')
            Pz = pd.read_csv(path,header=None)
            

        elif surface == 'Power Ellipsoid':
            path = os.path.join(dir_path, 'data/power_ellipsoid/eigenvalues.csv')
            eigenvalues = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/power_ellipsoid/eigenvectors.csv')
            eigenvectors = pd.read_csv(path,header=None)
            # load surface coordinate
            path = os.path.join(dir_path, 'data/power_ellipsoid/Px.csv')
            Px = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/power_ellipsoid/Py.csv')
            Py = pd.read_csv(path,header=None)
            path = os.path.join(dir_path, 'data/power_ellipsoid/Pz.csv')
            Pz = pd.read_csv(path,header=None)
            

        self.evs = eigenvalues.to_numpy().flatten()[1:]
        self.EVs = eigenvectors.to_numpy()[:,1:]

        # build 3D vector 
        P = np.column_stack((Px,Py))
        P = np.column_stack((P,Pz))
        self.P = P

        pos_idx, neg_idx = split_into_pos_neg_z_values(P)
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx 
        self.pos_pts = P[pos_idx]
        self.neg_pts = P[neg_idx]
        self.max_odd_idx = self.find_max_amplitude_idx_odd_overtones()
        self.max_even_idx = self.find_max_amplitude_idx_even_overtones()
        self.initial_idxs = zip_arrays(self.max_even_idx, self.max_odd_idx)

        

    def find_max_amplitude_idx_odd_overtones(self):
        pos_idx = self.pos_idx
        pos_EVs = self.EVs[pos_idx]
        #maxi = np.max(np.abs(pos_EVs), axis=0)
        max_idx = np.argmax(np.abs(pos_EVs), axis=0)
        odd_indices = np.arange(1, len(max_idx), 2)
        return max_idx[odd_indices]

    def find_max_amplitude_idx_even_overtones(self):
        neg_idx = self.neg_idx
        neg_EVs = self.EVs[neg_idx]
        #maxi = np.max(np.abs(neg_EVs), axis=0)
        max_idx = np.argmax(np.abs(neg_EVs), axis=0)
        even_indices = np.arange(0,len(max_idx), 2)
        return max_idx[even_indices]

    def find_center_point(self, pos_idx):
        P_xy = self.P[:, :2]
        P_xy = P_xy[pos_idx]
        norms = np.linalg.norm(P_xy, axis=1)
        return np.argmin(norms)

    def find_point_with_max_dist(self, pos_idx):
        X = self.P[:,0]
        return np.argmax(X)

    #def find_cone_center(self, )



S = Surface('Minor Ellipsoid')
#print(f"max_idx = {max_idx}")
print(f'odd_idices = {S.max_odd_idx}')
print(f'even_idices = {S.max_even_idx}')
zipped = zip_arrays(S.max_even_idx,S.max_odd_idx)
print(f'zipped = {S.initial_index}')
print(f'length = {len(S.initial_index)}')
#print(f"maxi = {maxi}")
#print(f'maxi.shape = {maxi.shape}')
a =  np.arange(5)
a += 1
b = np.arange(4)
c = zip(a,b)
#print(list(c))



arr_even = [2, 4, 6, 8]
arr_odd = [1, 3, 5, 7]
result = zip_arrays(arr_even, arr_odd)
#print(result)  # Output: [2, 1, 4, 3, 6, 5, 8, 7] 

