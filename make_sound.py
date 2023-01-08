import numpy as np
import pandas as pd
from initial_conditions import provide_initial_condition
from thinkdsp import CosSignal, SinSignal
from scipy import linalg
import time
import pathlib
import os

# hallo
class Make_Sound:
    def __init__(self, surface):

        cwd = os.getcwd() 
        print('Hallo')
        print(cwd)

        if surface == 'Minor Ellipsoid':
            eigenvalues = pd.read_csv('../data/minor_ellipsoid/eigenvalues.csv',header=None)
            #eigenvalues = eigenvalues.to_numpy()[1:]
            eigenvectors = pd.read_csv('../data/minor_ellipsoid/eigenvectors.csv',header=None)
            # load surface coordinate
            Px = pd.read_csv('../data/minor_ellipsoid/Px.csv',header=None)
            Py = pd.read_csv('../data/minor_ellipsoid/Py.csv',header=None)
            Pz = pd.read_csv('../data/minor_ellipsoid/Pz.csv',header=None)

        elif surface == 'Major Ellipsoid':
            eigenvalues = pd.read_csv('../data/major_ellipsoid/eigenvalues.csv',header=None)
            eigenvectors = pd.read_csv('../data/major_ellipsoid/eigenvectors.csv',header=None)
            # load surface coordinate
            Px = pd.read_csv('../data/major_ellipsoid/Px.csv',header=None)
            Py = pd.read_csv('../data/major_ellipsoid/Py.csv',header=None)
            Pz = pd.read_csv('../data/major_ellipsoid/Pz.csv',header=None)

        elif surface == 'Power Ellipsoid':
            eigenvalues = pd.read_csv('/Users/seidel/Desktop/repos/Surface_Sound/data/power_ellipsoid/eigenvalues.csv',header=None)
            eigenvectors = pd.read_csv('/Users/seidel/Desktop/repos/Surface_Sound/data/power_ellipsoid/eigenvectors.csv',header=None)
            # load surface coordinate
            Px = pd.read_csv('/Users/seidel/Desktop/repos/Surface_Sound/data/power_ellipsoid/Px.csv',header=None)
            Py = pd.read_csv('/Users/seidel/Desktop/repos/Surface_Sound/data/power_ellipsoid/Py.csv',header=None)
            Pz = pd.read_csv('/Users/seidel/Desktop/repos/Surface_Sound/data/power_ellipsoid/Pz.csv',header=None)
            

        self.evs = eigenvalues.to_numpy().flatten()[1:]
        self.EVs = eigenvectors.to_numpy()[:,1:]

        # build 3D vector
        P = np.column_stack((Px,Py))
        P = np.column_stack((P,Pz))
        self.P = P
        self.a = None
        self.initial_index = None
        

    def set_pick_matrix(self, initial_index, number_of_eigenvalues):
        EVs = self.EVs
        n = initial_index.shape[0]
        X = np.zeros([n,n])
        for i in range(number_of_eigenvalues):
            X[:,i] = EVs[:,i][initial_index]
        return X

    def set_beat_matrix(self, initial_index, c, number_of_eigenvalues):
        EVs = self.EVs
        w = self.evs
        n = initial_index.shape[0]
        X = np.zeros([n,n])
        for i in range(number_of_eigenvalues):
            X[:,i] = c * w[i] * EVs[:,i][initial_index]
        return X

   
    
    def determine_coefficients(self, EVs, evs, P, Start_P, propagation_velocity, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='pick'):
        i_f = initial_func
        p_o_b = pick_or_beat
        hight = 2
        # load initial conditions
        initial_condition, initial_index = provide_initial_condition(P, Start_P, hight, number_of_eigenvalues, initial_func=i_f)
        X = np.zeros([2,2])
        c = propagation_velocity
        w = evs

        if p_o_b == 'pick':
            X = self.set_pick_matrix(initial_index, number_of_eigenvalues)
        else:
            X = self.set_beat_matrix(initial_index, c, number_of_eigenvalues)
        print('initial = ' + str(initial_condition))
        print('X = ' + str(X))
        a = np.linalg.solve(X, initial_condition)
        print('a = ' + str(a))
        return a, initial_index

    
    def sound_single_vertex_pick(self, a, w, EVs, c, j, number_of_eigenvalues):
        sig = 0
        #print('first a = ' + str(a[0]))
        for i in range(number_of_eigenvalues):
            cos_sig = CosSignal(freq=c*w[i], amp=a[i]*EVs[j][i], offset=0)
            sig += cos_sig 
        return sig 

    def sound_single_vertex_beat(self, a, w, EVs, c, j, number_of_eigenvalues):
        sig = 0
        #print('first a = ' + str(a[0]))
        for i in range(number_of_eigenvalues):
            cos_sig = SinSignal(freq=c*w[i], amp=a[i]*EVs[j][i], offset=0)
            sig += cos_sig 
        return sig 


    def sound(self, a, w, EVs, c, initial_index, number_of_eigenvalues, pick_or_beat):
        signal = 0
        for i in initial_index:
            if pick_or_beat == 'pick':
                signal += self.sound_single_vertex_pick(a, w, EVs, c, i, number_of_eigenvalues)
            else:
                signal += self.sound_single_vertex_beat(a, w, EVs, c, i, number_of_eigenvalues)
        return signal

    
    def write_sound_to_file(self, c, number_of_eigenvalues, surface='power_ellipsoid', initial_func='sawtooth', pick_or_beat='pick'):
        EVs = self.EVs 
        evs = self.evs
        P = self.P
        Start_P = P[0]
        i_f = initial_func
        p_o_b = pick_or_beat
        a, initial_index = self.determine_coefficients(EVs, evs, P, Start_P, c, number_of_eigenvalues, initial_func=i_f, pick_or_beat=p_o_b)
        print('initial idx = ' + str(initial_index))
        print('first eigenvalue = ' + str(evs[1]))
        #f = self.sound(a, evs, EVs, c, initial_index, number_of_eigenvalues, pick_or_beat)
        if pick_or_beat == 'pick':
            f = self.sound_single_vertex_pick(a, evs, EVs, c, 0, number_of_eigenvalues)
        else:
            f = self.sound_single_vertex_beat(a, evs, EVs, c, 0, number_of_eigenvalues)
        wave = f.make_wave(duration=10.5, start=0, framerate=11025)
        audio = wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "new_signal.wav")

        open(filename, 'w').close()
        wave.write(filename)
        print('new signal written')
    
    
if __name__ == "__main__":
    ms = Make_Sound('Minor Ellipsoid')
    c = 0.3
    number_of_eigenvalues = 5
    Start_P = ms.P[0]
   #ms.determine_coefficients(Start_P, c, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='pick')
    ms.write_sound_to_file(c, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='beat')
    pass
