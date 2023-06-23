import numpy as np 
import surface as sf 
import os
import math
import matplotlib.pyplot as plt
from thinkdsp import CosSignal, SinSignal, normalize, decorate
import thinkdsp
import morphing

PI = math.pi 

def zip_arrays(arr_even, arr_odd):
    """
    zips even and odd array such that the values of the even have even indices and the odd hve odd indices
    """
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



class Make_Sound:
    def __init__(self, S, number_of_overtones, pick_or_beat):
        self.S = S 
        self.number_of_overtones = number_of_overtones
        self.pick_or_beat = pick_or_beat

    def set_matrix(self, c):
        """ 
        This gives the matrix to pull on the surface

        initial_index: the index of the vertex where we pull/pick

        number_of_eigenvalues: the number of overtones
        """
        s = self.S
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

    def set_cone_func(self, point, wedges_dict, h, n):
        wedge = find_wedge(point, n)
        max_point = wedges_dict.get(wedge)
        max_dist = np.linalg.norm(max_point)
        point_dist = np.linalg.norm(point)
        point_h = h * (max_dist - point_dist) / max_dist 
        return point_h

    def set_sawtooth_func(self, point, wedges_dict, h, n):
        wedge = find_wedge(point, n)
        max_point = wedges_dict.get(wedge)
        max_dist = np.linalg.norm(max_point)
        point_dist = np.linalg.norm(point)
        point_h = h * point_dist / max_dist 
        return point_h



    def set_initial_function(self, n, peak_range, initial_func):
        hight = 10.
        S = self.S 
        center = S.find_center_point()
        x_0 = center[0]
        x_1 = S.find_point_with_max_dist()
        x = (1 - peak_range) * x_0 + peak_range * x_1
        center[0] = x 
        P_xy = S.pos_pts[:,:2] - center
        # This should be a single statement
        wedge_arr = sf.set_wedge_array(P_xy, n)
        grouped_array = sf.group_points(P_xy, wedge_arr)
        max_norm_points, wedge_values = sf.find_max_norm_points(grouped_array, n)
        wedges_dict = dict(list(zip(wedge_values, max_norm_points)))

        selected_points = S.pos_pts[S.max_odd_idx][:,:2] - center
        if initial_func == 'cone':
            vfunc = np.vectorize(self.set_cone_func, otypes=[np.ndarray],signature="(n),(),(),() -> ()")
            hights = vfunc(selected_points, wedges_dict, hight, n)
            print(f'in cone')
        elif initial_func == 'sawtooth':
            print(f'in sawtooth')
            vfunc = np.vectorize(self.set_sawtooth_func, otypes=[np.ndarray],signature="(n),(),(),() -> ()")
            hights = vfunc(selected_points, wedges_dict, hight, n)
        else:
            print(f'in single point')
            hights = np.zeros(len(selected_points))
            hights[0] = hight
            
        ### Plot
        S = self.S
        P_xy = S.pos_pts[:,:2]
        x = selected_points[:,0]
        y = selected_points[:,1]
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.stem(x, y, hights)
        #plt.show()
        l = len(S.initial_idxs) - len(hights)
        zeros = np.zeros(l)
        initial_condition = zip_arrays(hights, zeros)
        return initial_condition 

    def determine_coefficients(self,peak_range, propagation_velocity, initial_func, pick_or_beat):
        """ 
        provide the coefficients of the generalized Fourier series
        """
        S = self.S 
        EVs = S.EVs
        evs = S.evs
        i_f = initial_func
        p_o_b = pick_or_beat
        n = 3
        #
        # load initial conditions
        initial_condition = self.set_initial_function(n, peak_range, initial_func)
        
        #X = np.zeros([2,2])
        c = propagation_velocity
        w = evs

        X = self.set_matrix(c)
        print(f'X = {X.shape}')
        a = np.linalg.solve(X, initial_condition)
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
        #print(x_spec)
        return x_spec, y_spec

    def get_spectrum(self, c, peak_range, number_of_eigenvalues, initial_func, pick_or_beat):
        S = self.S
        EVs = S.EVs 
        evs = S.evs
        print(f'P.shape = {S.P.shape[0]}')
        initial_index = S.initial_idxs
        #P = self.P
        #Start_P = P[0]
        i_f = initial_func
        p_o_b = pick_or_beat
        x_spec = np.zeros(number_of_eigenvalues)
        y_spec = np.zeros(number_of_eigenvalues)
        n = 3
        a = self.determine_coefficients(peak_range, c, i_f, p_o_b)
        print(f'initial function = {initial_func}')
        for i in range(S.P.shape[0]):
        #for i in initial_index:
            #print("")
            x_spec_single, y_spec_single = self.get_spectrum_single_vertex(a, c, i, number_of_eigenvalues)
            #print(y_spec_single) 
            if i == initial_index[0]:
                x_spec = np.add(x_spec,x_spec_single)
            y_spec = np.add(y_spec,np.abs(y_spec_single))
        #print(x_spec)
        #x_spec, y_spec = self.get_spectrum_single_vertex(a, c, 0, number_of_eigenvalues)
        print(f'y_spec = {y_spec}')
        return x_spec, y_spec

    def write_sound_to_file(self, c, peak_range, number_of_eigenvalues, initial_func, pick_or_beat):
        i_f = initial_func
        p_o_b = pick_or_beat
        x_spec, y_spec = self.get_spectrum(c,peak_range,number_of_eigenvalues,initial_func=i_f, pick_or_beat=p_o_b)
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

    def get_highest_frequency(self, c, number_of_eigenvalues):
        i = len(self.S.evs)
        return c*self.S.evs[number_of_eigenvalues-1]

    def set_random_spectrum(self, c, number_of_eigenvalues):
        hf = self.get_highest_frequency(c,number_of_eigenvalues)
        x_spec = np.random.uniform(0, hf * 1.2, number_of_eigenvalues)
        x_spec = np.sort(x_spec)
        y_spec = np.random.uniform(0, 1., number_of_eigenvalues)
        return x_spec, y_spec

    def create_morphing_func(self, c, number_of_eigenvalues, wave,p, initial_func='sawtooth', pick_or_beat='pick'):
        i_f = initial_func
        p_o_b = pick_or_beat
        n = len(wave.ys)
        d = 1 / wave.framerate
        spectrum_domain = np.fft.fftfreq(n,d)
        framerate = wave.framerate
        x_spec , y_spec = self.get_spectrum(c, peak_range, number_of_eigenvalues, i_f, p_o_b)
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
        len_rec = rec_spectrum.hs[0]
        convolution = self.create_morphing_func(c, number_of_eigenvalues, wave, p, i_f, p_o_b)
        len_conv = convolution[0]
        morphed_spectrum_hs = convolution * rec_spectrum.hs
        len_morph = morphed_spectrum_hs[0]
        ys = np.fft.ifft(morphed_spectrum_hs)
        morphed_spectrum = thinkdsp.Spectrum(morphed_spectrum_hs, rec_spectrum.fs, wave.framerate*2)
        #morphed_spectrum = thinkdsp.Spectrum(rec_spectrum.hs, rec_spectrum.fs, wave.framerate*2)
        #morphed_spectrum = np.abs(morphed_spectrum)
        morphed_wave = morphed_spectrum.make_wave()
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
        #print('new signal written')
        return morphed_wave, len_rec, len_conv, len_morph

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
        #print('new signal written')
        return wave


S = sf.Surface('Minor Ellipsoid')
MS = Make_Sound(S,20,'pick')
X = MS.set_matrix(0.2)
initial_func = MS.set_initial_function(3, 0.0, initial_func='sawtooth')
#print(initial_func)
c = 0.2
peak_range = 0.1
a = MS.determine_coefficients(peak_range, c, initial_func='cone', pick_or_beat='pick')
print(f'a = {a}')
MS.get_spectrum(c,peak_range, 100, initial_func='cone', pick_or_beat='pick')
