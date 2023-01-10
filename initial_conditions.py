import numpy as np
import pandas as pd


def dist(P, Q):
    """euclidean distance between two points"""
    return np.linalg.norm(P-Q)


def calculate_dist(P, Start_P):
    """returns the euclidian distance from a selected point to all other points"""
    dist_func = np.vectorize(dist)
    dist_func = dist_func(P, Start_P)
    P0 = dist_func[:,0]
    P1 = dist_func[:,1]
    P2 = dist_func[:,2]
    return np.sqrt(P0*P0 + P1*P1 + P2*P2)

def calculate_used_index(sorted_dist_selection, index_dict):
    """calculate sawtooth initial condition"""
    used_index = []
    for d in sorted_dist_selection:
        used_index.append(index_dict[d])
    used_index = np.array(used_index)
    return used_index


def sawtooth_func(n, hight, sorted_dist_selection, used_index, dist_dict):
    radius = np.max(sorted_dist_selection)
    initial_condition = np.zeros(n)
    for i in used_index:
        initial_condition[i] = dist_dict[i] * hight / radius
    return initial_condition

def cone_func(n, hight, sorted_dist_selection, used_index, dist_dict):
    radius = np.max(sorted_dist_selection)
    initial_condition = np.zeros(n)
    for i in used_index:
        initial_condition[i] = (1 - dist_dict[i] / radius) * hight
    return initial_condition

def rectangle_anulus_func(n, hight, sorted_dist_selection, used_index, dist_dict):
    radius = np.max(sorted_dist_selection)
    initial_condition = np.zeros(n)
    for i in used_index:
        if  dist_dict[i] >= radius / 2:
            initial_condition[i] = hight
    return initial_condition

def rectangle_cylinder_func(n, hight, sorted_dist_selection, used_index):
    return np.ones(n) * hight

def provide_initial_condition(P, Start_P, hight, number_of_eigenvalues, initial_func=None):
    """
    Returns: function of initial conditions and the corresponding vertex index
    """
    dist_list = calculate_dist(P, Start_P)
    n = len(dist_list)
    index = np.arange(n)
    sorted_dist = dist_list.copy()
    sorted_dist = np.sort(sorted_dist)
    index_dict = dict(zip(dist_list,index))
    dist_dict = dict(zip(index,dist_list))
    num_sawtooth = int(number_of_eigenvalues / 2)
    #print('num_sawtoot = '+str(num_sawtooth))
    sorted_dist_selection = sorted_dist[0:num_sawtooth]
    used_index = calculate_used_index(sorted_dist_selection, index_dict)

    # select points randomly in the rest of the points
    rest_P = np.delete(P,used_index)
    rest_index = np.setdiff1d(index,used_index)
    #np.random.seed(10)
    rest_choice = np.random.choice(rest_index,size=number_of_eigenvalues - num_sawtooth)

    # calculate sawtooth initial condition
    if initial_func is None:
        initial_condition = cone_func(n, hight, sorted_dist_selection, used_index, dist_dict)
    elif initial_func == 'sawtooth':
        initial_condition = sawtooth_func(n, hight, sorted_dist_selection, used_index, dist_dict)
    elif initial_func == 'rectangle':
        initial_condition = rectangle_anulus_func(n, hight, sorted_dist_selection, used_index, dist_dict)
    elif initial_func == 'cylinder':
        initial_condition = rectangle_cylinder_func(n, hight, sorted_dist_selection, used_index)
    elif initial_func == 'cone':
        initial_condition = cone_func(n, hight, sorted_dist_selection, used_index)


    # add the choosen vetices from the resr of the surface
    initial_index = np.hstack((rest_choice,used_index))
    initial_index = np.sort(initial_index)
    return initial_condition[initial_index], initial_index
    

hight = 1
#initial_condition, initial_index = initial_condition(P, Start_P, hight, initial_func='sawtooth')
#print('new func')
#print(initial_index)
#print(initial_condition)