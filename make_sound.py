import numpy as np 
import surface as sf 
import os
import math
import matplotlib.pyplot as plt
from thinkdsp import CosSignal, SinSignal, normalize, decorate
import thinkdsp
import morphing
from collections import Counter
import scipy

PI = math.pi 

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
    wedges = set_wedge_array(points,3)
    expected_wedges = np.array([3,3,5])
    assert np.allclose(wedges, expected_wedges), "test_wedge_array(): do not match the expected values."
    

test_wedge_array()

def sum_up_multiple_eigenvalues(arr):
    # Use Counter to count occurrences of unique values
    counter = Counter(arr)   
    # Multiply the unique values by their counts
    new_array = np.array([value * count for value, count in counter.items()]) 
    return new_array




class Make_Sound:
    def __init__(self, S, number_of_overtones, pick_or_beat, x_spec=None, y_spec=None):
        self.S = S 
        self.number_of_overtones = number_of_overtones
        self.pick_or_beat = pick_or_beat
        self.x_spec = x_spec
        self.y_spec = y_spec

    

    def set_matrix(self, c):
        """ 
        This gives the matrix to pick or hit on the surface

        initial_index: the index of the vertex where we pick or hit

        number_of_eigenvalues: the number of overtones
        """
        S = self.S
        p_o_b = self.pick_or_beat
        EVs = S.EVs
        w = S.evs
        initial_index = S.initial_idxs
        n = initial_index.shape[0]
        X = np.zeros([n,n])
        if p_o_b == 'pick':
            for i in range(n):
                X[:,i] = EVs[:,i][initial_index]
        elif p_o_b == 'hit':
            for i in range(n):
                X[:,i] = c * w[i] * EVs[:,i][initial_index]
        return X


    def set_upper_cone_func(self, point, wedges_dict, h, n, l):
        wedge = find_wedge(point, n)
        max_point = wedges_dict.get(wedge)
        max_dist = np.linalg.norm(max_point)
        point_dist = np.linalg.norm(point)
        scaled_dist = point_dist / max_dist
        point_h = 0.0
        if l > scaled_dist:
            point_h = h * (1. - scaled_dist / 2. / l)
        return point_h

    def set_lower_cone_func(self, point, wedges_dict, h, n, l):
        wedge = find_wedge(point, n)
        max_point = wedges_dict.get(wedge)
        max_dist = np.linalg.norm(max_point)
        point_dist = np.linalg.norm(point)
        scaled_dist = point_dist / max_dist
        point_h = 0.0
        if (l - 0.5) > (1. - scaled_dist) / 2.:
            new_dist = (2. - scaled_dist) / 2.
            point_h =  h * (1. - new_dist / l)
        return point_h



    def set_upper_cylinder_func(self, point, wedges_dict, h, n, l):
        wedge = find_wedge(point, n)
        max_point = wedges_dict.get(wedge)
        max_dist = np.linalg.norm(max_point)
        point_dist = np.linalg.norm(point)
        scaled_dist = point_dist / max_dist
        point_h = 0.0
        if l > scaled_dist/ 2.:
            point_h = h
        return point_h
    
    def set_lower_cylinder_func(self, point, wedges_dict, h, n, l):
        wedge = find_wedge(point, n)
        max_point = wedges_dict.get(wedge)
        max_dist = np.linalg.norm(max_point)
        point_dist = np.linalg.norm(point)
        scaled_dist = point_dist / max_dist
        point_h = 0.0
        if (l - 0.5) > (1. - scaled_dist)/2.:
            point_h = h 
        return point_h
    
    def set_cylinder_func(self, point, z, wedges_dict, h, n, l):
        if z > 0:
            return self.set_upper_cylinder_func(point, wedges_dict, h, n, l)
        else:
            return self.set_lower_cylinder_func(point, wedges_dict, h, n, l)
      
    def set_cone_func(self, point, z, wedges_dict, h, n, l):
        if z > 0:
            return self.set_upper_cone_func(point, wedges_dict, h, n, l)
        else:
            return self.set_lower_cone_func(point, wedges_dict, h, n, l)
        
    def set_wedges_dict(self, P_xy, n):
        wedge_arr = sf.set_wedge_array(P_xy, n)
        grouped_array = sf.group_points(P_xy, wedge_arr)
        max_norm_points, wedge_values = sf.find_max_norm_points(grouped_array, n)
        wedges_dict = dict(list(zip(wedge_values, max_norm_points)))
        return wedges_dict


    def set_initial_function(self, n, l, initial_func):
        hight = 1.
        S = self.S 
        center = S.find_center_point()
        x_0 = center[0]
        x_1 = S.find_point_with_max_dist()
        x = (1 - l) * x_0 + l * x_1
        center[0] = x 
        P_xy = S.P[:,:2]
        P_z = S.P[:,2]
        
        wedges_dict = self.set_wedges_dict(P_xy, n)
        selected_points = P_xy[S.initial_idxs]
        hights = np.zeros(len(selected_points))
        selected_points = P_xy[S.initial_idxs]
        z = P_z[S.initial_idxs]
        if initial_func == 'Cone':
            vfunc = np.vectorize(self.set_cone_func, otypes=[np.ndarray],signature="(n),(),(),(),(),() -> ()")
            hights = vfunc(selected_points,z, wedges_dict, hight, n,l)
        elif initial_func == 'Cylinder':
            vfunc = np.vectorize(self.set_cylinder_func, otypes=[np.ndarray],signature="(n),(),(),(),(),() -> ()")
            hights = vfunc(selected_points,z, wedges_dict, hight, n, l)
            

        #graph_domain = np.copy(selected_points)
        #for i in range(len(graph_domain)):
        #    if z[i] <= 0:
        #        l_p = np.linalg.norm(graph_domain[i])
        #        graph_domain[i] = (2.*x_1/l_p - 1.) * graph_domain[i]   
        ## Plot
        #x = graph_domain[:,0]
        #y = graph_domain[:,1]
        #print(f'x = {x}')
        #print(f'y = {y}')
        #print(f'hights = {hights}')
        #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        #ax.stem(x, y, hights)
        #plt.show()
        return hights 

    def determine_coefficients(self,l, propagation_velocity, initial_func, pick_or_beat):
        """ 
        provide the coefficients of the generalized Fourier series
        """
        S = self.S 
        EVs = S.EVs
        evs = S.evs
        i_f = initial_func
        p_o_b = pick_or_beat
        n = 3
        # load initial conditions
        initial_condition = self.set_initial_function(n, l, initial_func)
        #initial_condition = initial_condition.astype('float64')
        c = propagation_velocity
        w = evs
        X = self.set_matrix(c)
        X = X.astype('float64')
        # Here I use Singular Value Decomposition (SVD): Instead of directly solving the system using np.linalg.solve 
        # in order to avoid singular value exception
        U, s, V = np.linalg.svd(X)
        a = np.dot(np.dot(V.T, np.linalg.inv(np.diag(s))), np.dot(U.T, initial_condition))
        return a

    def get_spectrum_single_vertex(self, a, c, j, number_of_eigenvalues):
        S = self.S
        w = S.evs
        EVs = S.EVs
        x_spec = np.zeros(number_of_eigenvalues)
        y_spec = np.zeros(number_of_eigenvalues)
        for i in range(number_of_eigenvalues):
            x_spec[i] = c*w[i]
            y_spec[i] = abs(a[i]*EVs[j][i])
        return x_spec, y_spec


    def y_spec_ij(self, i, j, a, c):
        EVs = self.S.EVs
        return abs(a[i]*EVs[j][i])


    def get_y_spec_single_vertex(self, j, a, c, number_of_eigenvalues):
        EVs = self.S.EVs
        y_spec = np.zeros(number_of_eigenvalues)

        i_range = np.arange(number_of_eigenvalues)
        vfunc = np.vectorize(self.y_spec_ij,otypes=[np.ndarray], signature="(m),(),(n),() -> (m)")
        y_spec = vfunc(i_range, j, a, c)
        return y_spec

    def get_x_spec(self, c, number_of_eigenvalues):
        w = self.S.evs
        x_spec = np.zeros(number_of_eigenvalues)
        for i in range(number_of_eigenvalues):
            x_spec[i] = c*w[i]
        return x_spec 

    def get_spectrum(self, c, initial_value_domain, number_of_eigenvalues, initial_func, pick_or_beat):
        S = self.S
        EVs = S.EVs 
        evs = S.evs
        initial_index = S.initial_idxs
        i_f = initial_func
        p_o_b = pick_or_beat
        x_spec = np.zeros(number_of_eigenvalues)
        y_spec = np.zeros(number_of_eigenvalues)
        n = 3
        a = self.determine_coefficients(initial_value_domain, c, i_f, p_o_b)
        #The next rows are simply:
        #for j in range(S.P.shape[0]):
        #    y_spec_single = self.get_y_spec_single_vertex(j, a, c, number_of_eigenvalues) 
        #    y_spec = np.add(y_spec,y_spec_single)   

        vfunc = np.vectorize(self.get_y_spec_single_vertex,  otypes=[np.ndarray], signature="(n),(m),(),() -> (n,o)")
        y_spec_single = vfunc(np.arange(S.P.shape[0]),a, c, number_of_eigenvalues)
        y_spec = np.sum(y_spec_single, axis=0)

        x_spec = self.get_x_spec(c, number_of_eigenvalues)
        # Get unique values and their indices
        unique_values, indices = np.unique(x_spec, return_index=True)
        # Sort the unique values based on their indices
        sorted_indices = np.argsort(indices)
        sorted_unique_values = unique_values[sorted_indices]
        # Create the new array with unique values
        x_new = sorted_unique_values
        # Sum up the corresponding y values for each unique value of x
        y_new = [np.sum(y_spec[x_spec == value]) for value in x_new]
        #y_spec = sum_up_multiple_eigenvalues(y_spec)
        self.x_spec = x_new
        self.y_spec = y_new
    

        

    def write_sound_to_file(self, c, initial_value_domain, initial_func, pick_or_beat):
        i_f = initial_func
        p_o_b = pick_or_beat
        self.get_spectrum(c,initial_value_domain,self.number_of_overtones,initial_func=i_f, pick_or_beat=p_o_b)
        x_spec = self.x_spec
        y_spec = self.y_spec
        sig = 0
        if pick_or_beat == 'pick':
            for i in range(len(x_spec)):
                sig += CosSignal(freq=x_spec[i], amp=y_spec[i], offset=0)
        else:
            for i in range(len(x_spec)):
                sig += SinSignal(freq=x_spec[i], amp=y_spec[i], offset=0)

        wave = sig.make_wave(duration=11.5, start=0, framerate=44100)
        audio = wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "new_signal.wav")

        open(filename, 'w').close()
        wave.write(filename)
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
        ts = np.linspace(0, 1, framerate)
        sig = self.synthesize(amps, freqs, ts)
        return sig 


    def sound_single_vertex_beat(self, a, w, EVs, c, j, number_of_eigenvalues):
        sig = None
        for i in range(number_of_eigenvalues):
            sin_sig = SinSignal(freq=c*w[i], amp=abs(a[i]*EVs[j][i]), offset=0)
            sig += sin_sig 
        return sig 

    def get_highest_frequency(self, c, number_of_eigenvalues):
        """

        """
        i = len(self.S.evs)
        return c*self.S.evs[number_of_eigenvalues-1]


    def create_morphing_func(self, c, wave, p):
        n = len(wave.ys)
        d = 1. / wave.framerate
        spectrum_domain = np.fft.fftfreq(n,d)
        framerate = wave.framerate
        x_spec = self.x_spec
        y_spec = self.y_spec
        gf = morphing.Global_Function(p, x_spec, y_spec, spectrum_domain)
        morphing.set_global_function(gf)
        global_func = gf.func
        length = len(spectrum_domain)
        convolution = morphing.smooth_func(global_func, p, length)
        return convolution

    def write_morphed_sound(self, c, wave,recorded_wave, p):
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
        len_rec = rec_spectrum.hs[0]
        convolution = self.create_morphing_func(c, wave, p)
        len_conv = convolution[0]
        morphed_spectrum_hs = convolution * rec_spectrum.hs
        len_morph = morphed_spectrum_hs[0]
        ys = np.fft.ifft(morphed_spectrum_hs)
        morphed_spectrum = thinkdsp.Spectrum(morphed_spectrum_hs, rec_spectrum.fs, wave.framerate*2)
        morphed_wave = morphed_spectrum.make_wave()
        morphed_wave.normalize()
        #morphed_wave = rec_spectrum.make_wave()

        audio = morphed_wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "morphed_signal.wav")

        open(filename, 'w').close()
        morphed_wave.write(filename)
        return morphed_wave

    def write_envelope_sound(self, c, wave, p,noise):
        spectrum = wave.make_spectrum(full=True)
        envelope = self.create_morphing_func(c, wave, p)
        white_signal = thinkdsp.PinkNoise(beta=2)
        if noise == 'White Noise':
            white_signal = thinkdsp.UncorrelatedUniformNoise()
        elif noise == 'Brownian Noise':
            white_signal = thinkdsp.BrownianNoise()
        elif noise == 'Pink Noise':
            white_signal = thinkdsp.PinkNoise(beta=2)
        
        white_wave = white_signal.make_wave(duration=11.5, start=0, framerate=44100)
        l1 = wave.__len__()
        l2 = white_wave.__len__()
        length = l1
        if l1 > l2:
            length = l2
            wave.truncate(length)
        else:
            white_wave.truncate(length)

        white_spectrum = white_wave.make_spectrum(full=True)

        envelope = envelope * white_spectrum.hs
        len_conv = envelope[0]
        envelope_spectrum = thinkdsp.Spectrum(envelope, spectrum.fs, wave.framerate,full=True)
        envelope_wave = envelope_spectrum.make_wave()
        envelope_wave.normalize()
        #morphed_wave = rec_spectrum.make_wave()

        audio = envelope_wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "envelope_signal.wav")

        open(filename, 'w').close()
        envelope_wave.write(filename)
        return envelope_wave

    def set_random_spectrum(self, c, number_of_eigenvalues):
        """
        This generates a discrete random spactrum
        """
        hf = self.get_highest_frequency(c,number_of_eigenvalues)
        x_spec = np.random.uniform(0, hf * 1.2, number_of_eigenvalues)
        x_spec = np.sort(x_spec)
        y_spec = np.random.uniform(0, 1., number_of_eigenvalues)
        return x_spec, y_spec

    def write_random_sound_to_file(self, c,  number_of_eigenvalues):
        x_spec, y_spec = self.set_random_spectrum(c, number_of_eigenvalues)
        sig = 0
        for i in range(len(x_spec)):
                sig += SinSignal(freq=x_spec[i], amp=y_spec[i], offset=0)

        wave = sig.make_wave(duration=5.5, start=0, framerate=44100)
        audio = wave.make_audio()

        # Get the current user's home directory
        home = os.path.expanduser("~")

        # Set the path to the desktop
        desktop = os.path.join(home, "Desktop")

        # Set the full path to the file
        filename = os.path.join(desktop, "random_signal.wav")

        open(filename, 'w').close()
        wave.write(filename)
        return wave


#S = sf.Surface('Power Ellipsoid')
#P = S.P 
#EVs = S.EVs
#MS = Make_Sound(S,20,'pick')
#X = MS.set_matrix(0.2)

#initial_func = MS.set_initial_function(3, 0.9, initial_func='cone')
#c = 0.2
#l = 0.1
#a = MS.determine_coefficients(l, c, initial_func='cylinder', pick_or_beat='pick')
#MS.write_sound_to_file( c, l, initial_func='Cylinder', pick_or_beat='pick')
