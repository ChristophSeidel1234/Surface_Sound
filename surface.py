import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

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

def group_points(points, wedge_arr):
    grouped_array = []
    unique_values = np.unique(wedge_arr)
    for value in unique_values:
        group = points[wedge_arr == value]
        grouped_array.append(group)
    return grouped_array

def find_max_norm_points(grouped_array, n):
    max_norm_points = []
    wedge_values = []
    
    for group in grouped_array:
        group_points = np.array(group)
        norms = np.linalg.norm(group_points, axis=1)
        max_norm_index = np.argmax(norms)
        max_norm_point = group_points[max_norm_index]
        max_norm_points.append(max_norm_point)
        
        wedge_value = find_wedge(max_norm_point, n)  # Assuming find_wedge is a function that returns the wedge value
        wedge_values.append(wedge_value)
    
    return max_norm_points, wedge_values

def set_cone_func(point, wedges_dict, h, n):
    wedge = find_wedge(point, n)
    max_point = wedges_dict.get(wedge)
    max_dist = np.linalg.norm(max_point)
    point_dist = np.linalg.norm(point)
    point_h = h * (max_dist - point_dist) / max_dist 
    return point_h

def set_sawtooth_func(point, wedges_dict, h, n):
    wedge = find_wedge(point, n)
    max_point = wedges_dict.get(wedge)
    max_dist = np.linalg.norm(max_point)
    point_dist = np.linalg.norm(point)
    point_h = h * point_dist / max_dist 
    return point_h

def set_anulus_func():
    return 0


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
        self.max_even_idx = self.find_max_amplitude_idx_even_overtones()
        self.max_odd_idx = self.find_max_amplitude_idx_odd_overtones()
        
        #self.initial_idxs = np.array(zip_arrays(self.max_even_idx, self.max_odd_idx))
        self.initial_idxs = np.argmax(np.abs(self.EVs), axis=0)

    def set_initial_idxs(self,random_vertices):
        initial_idxs = None
        if random_vertices:
            length = len(self.P)
            self.initial_idxs = np.random.choice(length, 100)
        else:
            #self.initial_idxs = np.array(zip_arrays(self.max_even_idx, self.max_odd_idx))
            self. initial_idxs = np.argmax(np.abs(self.EVs), axis=0)

    def find_max_amplitude_idx_overtones(self):
        return np.argmax(np.abs(self.EVs), axis=0)

         

    def find_max_amplitude_idx_odd_overtones(self):
        """
        These are the verices that receive corresponding values in the initial conditions.
        """
        neg_idx = self.neg_idx
        neg_EVs = self.EVs[neg_idx]
        max_idx = np.argmax(np.abs(neg_EVs), axis=0)
        odd_idices = np.arange(1,len(max_idx), 2)
        return max_idx[odd_idices]

        

    def find_max_amplitude_idx_even_overtones(self):
        """
        These are the verices that are set to zero in the initial conditions.
        """
        pos_idx = self.pos_idx
        pos_EVs = self.EVs[pos_idx]
        max_idx = np.argmax(np.abs(pos_EVs), axis=0)
        even_idices = np.arange(0, len(max_idx), 2)
        return max_idx[even_idices]

    def find_center_point(self):
        P_xy = self.pos_pts[:,:2]
        l = float(len(P_xy))
        center = np.sum(P_xy, axis=0)/l
        return center

    def find_point_with_max_dist(self):
        X = self.pos_pts[:,0]
        return np.max(X)

    def set_initial_function(self, n, peak_range, initial_func='sawtooth'):
        hight = 1.
        center = self.find_center_point()
        #print(f'center = {center}')
        x_0 = center[0]
        x_1 = self.find_point_with_max_dist()
        x = (1 - peak_range) * x_0 + peak_range * x_1
        center[0] = x 
        P_xy = self.P[:,:2] - center
        # This should be a single statement
        wedge_arr = set_wedge_array(P_xy, n)
        grouped_array = group_points(P_xy, wedge_arr)
        max_norm_points, wedge_values = find_max_norm_points(grouped_array, n)
        wedges_dict = dict(list(zip(wedge_values, max_norm_points)))

        selected_points = self.pos_pts[self.max_even_idx][:,:2] - center
        if initial_func == 'cone':
            vfunc = np.vectorize(set_cone_func, otypes=[np.ndarray],signature="(n),(),(),() -> ()")
        elif initial_func == 'sawtooth':
            vfunc = np.vectorize(set_sawtooth_func, otypes=[np.ndarray],signature="(n),(),(),() -> ()")


        hights = vfunc(selected_points, wedges_dict, hight, n)
        ### Plot
        P_xy = self.pos_pts[:,:2]
        x = selected_points[:,0]
        y = selected_points[:,1]
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.stem(x, y, hights)
        #plt.show()
        return hights

#def

S = Surface('Power Ellipsoid')
S.set_initial_function( 3, 0., initial_func='sawtooth')
#print(f"max_idx = {max_idx}")
#print(f'odd_idices = {S.max_odd_idx}')
#print(f'even_idices = {S.max_even_idx}')
zipped = zip_arrays(S.max_even_idx,S.max_odd_idx)
#print(f'zipped = {S.initial_idxs}')
#print(f'length = {len(S.initial_idxs)}')
points = S.pos_pts[:,:2]
#print(f'first point = {points[0]}')
#print(f'wedge = {find}')
n = 3
wedge_arr = set_wedge_array(points, n)

grouped_array = group_points(points, wedge_arr)
first_point = grouped_array[0][0]
#print(f'first point = {grouped_array[0][0]}')
first_wedge = find_wedge(first_point,3)

max_norm_points, wedge_values = find_max_norm_points(grouped_array,n)

for p in max_norm_points:
    wedge = find_wedge(p, n)
    #print(f'wedge = {wedge}')



