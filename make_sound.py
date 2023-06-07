import numpy as np
import pandas as pd
from initial_conditions import provide_initial_condition
from thinkdsp import CosSignal, SinSignal, normalize, decorate
import thinkdsp
from scipy import linalg
#import time
import os
import morphing


class Make_Sound:
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
        self.a = None
        self.initial_index = None


    def synthesize_cos(self, amps, fs, ts):
        """
        This makes a generalized cosine Fourier series from the frequencies and the amplitudes

        amps: array of amplitudes

        fs: array of frequencies
        """
        framerate = 11025
        components = [thinkdsp.CosSignal(freq, amp)
                    for amp, freq in zip(amps, fs)] 
        signal = thinkdsp.SumSignal(*components)
        ys = signal.evaluate(ts) 
        return ys
        

    def set_pick_matrix(self, initial_index, number_of_eigenvalues):
        """ 
        This gives the matrix to pull on the surface

        initial_index: the index of the vertex where we pull/pick

        number_of_eigenvalues: the number of overtones
        """
        EVs = self.EVs
        n = initial_index.shape[0]
        X = np.zeros([n,n])
        for i in range(number_of_eigenvalues):
            X[:,i] = EVs[:,i][initial_index]
        return X

    def set_beat_matrix(self, initial_index, c, number_of_eigenvalues):
        """ 
        This gives the matrix to hit the surface

        initial_index: the index of the vertex where we hit/beat

        number_of_eigenvalues: the number of overtones
        """
        EVs = self.EVs
        w = self.evs
        n = initial_index.shape[0]
        X = np.zeros([n,n])
        for i in range(number_of_eigenvalues):
            X[:,i] = c * w[i] * EVs[:,i][initial_index]
        return X

    
    def determine_coefficients(self, EVs, evs, P, Start_P, propagation_velocity, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='pick'):
        """ 
        provide the coefficients of the generalized Fourier series
        """
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
        a = np.linalg.solve(X, initial_condition)
        return a, initial_index

    def get_spectrum_single_vertex(self, a, w, EVs, c, j, number_of_eigenvalues):
        x_spec = np.zeros(number_of_eigenvalues)
        y_spec = np.zeros(number_of_eigenvalues)
        for i in range(number_of_eigenvalues):
            x_spec[i] = c*w[i]
            y_spec[i] = abs(a[i]*EVs[j][i])
        #print(x_spec)
        return x_spec, y_spec
    
    def get_spectrum(self, c, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='pick'):
        EVs = self.EVs 
        evs = self.evs
        P = self.P
        Start_P = P[0]
        i_f = initial_func
        p_o_b = pick_or_beat
        x_spec = np.zeros(number_of_eigenvalues)
        y_spec = np.zeros(number_of_eigenvalues)
        a, initial_index = self.determine_coefficients(EVs, evs, P, Start_P, c, number_of_eigenvalues, initial_func=i_f, pick_or_beat=p_o_b)
        for i in initial_index:
            print("")
            x_spec_single, y_spec_single = self.get_spectrum_single_vertex(a, evs, EVs, c, i, number_of_eigenvalues)
            #print(x_spec_single) 
            if i == 0:
                x_spec = np.add(x_spec,x_spec_single)
            y_spec = np.add(y_spec,y_spec_single)
        #print(x_spec)
        x_spec, y_spec = self.get_spectrum_single_vertex(a, evs, EVs, c, 1, number_of_eigenvalues)
        
        return x_spec, y_spec

    def write_sound_to_file(self, c, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='pick'):
        i_f = initial_func
        p_o_b = pick_or_beat
        x_spec, y_spec = self.get_spectrum(c,number_of_eigenvalues,initial_func=i_f, pick_or_beat=p_o_b)
        sig = 0
        if pick_or_beat == 'pick':
            for i in range(len(x_spec)):
                sig += CosSignal(freq=x_spec[i], amp=y_spec[i], offset=0)
        else:
            for i in range(len(x_spec)):
                sig += SinSignal(freq=x_spec[i], amp=y_spec[i], offset=0)

        wave = sig.make_wave(duration=5.5, start=0, framerate=44100)
        audio = wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "new_signal.wav")

        open(filename, 'w').close()
        wave.write(filename)
        #print('new signal written')
        return wave

    
    def sound_single_vertex_pick(self, a, w, EVs, c, j, number_of_eigenvalues):
        sig = 0
        for i in range(number_of_eigenvalues):
            cos_sig = CosSignal(freq=c*w[i], amp=a[i]*EVs[j][i], offset=0)
            sig += cos_sig 
        sig.plot()
        return sig 

    
    def sound_single_vertex_pick1(self, a, w, EVs, c, j, number_of_eigenvalues):
        amps = np.zeros(number_of_eigenvalues)
        freqs = np.zeros(number_of_eigenvalues)
        framerate = 11025

        for i in range(number_of_eigenvalues):
            amps[i] = a[i]*EVs[j][i]
            freqs[i] = c*w[i]
        high, low = abs(max(amps)), abs(min(amps))
        amps = amps / max(high, low)
        print('frequencies')
        print(freqs)
        print(amps)
        ts = np.linspace(0, 1, framerate)
        sig = self.synthesize(amps, freqs, ts)
        print('Here comes the signal')
        #sig.plot()
        return sig 

    

    def sound_single_vertex_beat(self, a, w, EVs, c, j, number_of_eigenvalues):
        sig = None
        #print('first a = ' + str(a[0]))
        for i in range(number_of_eigenvalues):
            sin_sig = SinSignal(freq=c*w[i], amp=abs(a[i]*EVs[j][i]), offset=0)
            sig += sin_sig 
        #sig.plot()
        return sig 




    def sound(self, a, w, EVs, c, initial_index, number_of_eigenvalues, pick_or_beat):
        signal = 0
        for i in initial_index:
            if pick_or_beat == 'pick':
                signal += self.sound_single_vertex_pick(a, w, EVs, c, i, number_of_eigenvalues)
            else:
                signal += self.sound_single_vertex_beat(a, w, EVs, c, i, number_of_eigenvalues)
        signal = normalize(signal)
        signal.plot()
        decorate(xlabel='Time (s)')
        return signal


    def get_highest_frequency(self, c, number_of_eigenvalues):
        i = len(self.evs)
        return c*self.evs[number_of_eigenvalues-1]

    def create_morphing_func(self, c, number_of_eigenvalues, wave,p, initial_func='sawtooth', pick_or_beat='pick'):
        i_f = initial_func
        p_o_b = pick_or_beat
        n = len(wave.ys)
        d = 1 / wave.framerate
        spectrum_domain = np.fft.fftfreq(n,d)
        framerate = wave.framerate
        x_spec , y_spec = self.get_spectrum( c, number_of_eigenvalues, i_f, p_o_b)
        gf = morphing.Global_Function(p, x_spec, y_spec, spectrum_domain)
        morphing.set_global_function(gf)
        global_func = gf.func
        length = len(spectrum_domain)
        convolution = morphing.smooth_func(global_func, p, length)
        return convolution

    def write_morphed_sound(self, c, number_of_eigenvalues, wave,recorded_wave,p, initial_func='sawtooth', pick_or_beat='pick'):
        i_f = initial_func
        p_o_b = pick_or_beat
        l1 = wave.__len__()
        l2 = recorded_wave.__len__()
        length = l1
        if l1 > l2:
            length = l2
            wave.truncate(length)
        else:
            recorded_wave.truncate(length)
        
        sectrum = wave.make_spectrum(full=True)
        rec_spectrum = recorded_wave.make_spectrum(full=True)
        #spectrum_domain = rec_spectrum.fs
        #spectrum_hs = rec_spectrum.hs
        convolution = self.create_morphing_func(c, number_of_eigenvalues, wave, p, i_f, p_o_b)
        morphed_spectrum_hs = convolution * rec_spectrum.hs
        morphed_spectrum = thinkdsp.Spectrum(morphed_spectrum_hs, rec_spectrum.fs, wave.framerate)
        morphed_wave = morphed_spectrum.make_wave()

        audio = morphed_wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "morphed_signal.wav")

        open(filename, 'w').close()
        morphed_wave.write(filename)
        #print('new signal written')
        return morphed_wave


    
    
if __name__ == "__main__":
    ms = Make_Sound('Minor Ellipsoid')
    c = 0.3
    number_of_eigenvalues = 5
    Start_P = ms.P[0]
   #ms.determine_coefficients(Start_P, c, number_of_eigenvalues, initial_func='sawtooth', pick_or_beat='pick')
    ms.write_sound_to_file(c, number_of_eigenvalues, initial_func='cone', pick_or_beat='beat')
    pass
