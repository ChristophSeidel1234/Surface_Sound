import numpy as np 
import math
import scipy.signal
import matplotlib.pyplot as plt
import thinkdsp as dsp
from thinkdsp import SinSignal, CosSignal




def rectangle(t,x_spec,y_spec,p):
    """
    Constructs a rectangular function that takes the value y_spec on the interval (x_spec - p,x_spec + p] and is zero otherwise.

    Args:
        t (float): parameter
        x_spec (float): position where the spectrum is not zero.
        y_spec (float): spectrum value.
        p (float): distance from x_spec.

    Returns:
        int: rectangle function.
    """
    if t <= x_spec - p or t > x_spec + p:
        return 0.0
    else:
        return y_spec

def make_y_spectrum(x_spec, y_spec, domain):
    indices = np.abs(domain - x_spec[:, None]).argmin(axis=1)
    image = np.zeros(len(domain))
    image[indices] = y_spec
    return image

def make_spectrum(x_spec, y_spec, domain,framerate):
        image = make_y_spectrum(x_spec, y_spec, domain)
        Spectrum = dsp.Spectrum(image,domain,framerate, full=True)
        return Spectrum

class Rectangle_Function:

    def __init__(self, p, x, y,domain):
        self.p = p
        self.x = x
        self.y = y
        self.domain = domain
        self.rectangle = None
        if domain.size > 0:
            vfunc = np.vectorize(rectangle)
            self.rectangle = vfunc(domain, x, y, p)
                

    def to_string(self):
        return f"Rectangle_Function\n={self.rectangle}"

  

class Global_Function:

    def __init__(self, p, x_spec, y_spec, domain):
        self.p = p
        self.x_spec = x_spec
        self.y_spec = y_spec
        self.domain = domain
        self.max_spec = np.amax(y_spec)
        self.max_idx = np.argmax(y_spec)
        self.x_value =  x_spec[self.max_idx]
        self.func = np.zeros(domain.size)
    


    #def to_string(self):
    #    return f"Global_Function\np={self.p}\nx_spec={self.x_spec}\ny_spec={self.y_spec}\ndomain={self.domain}\n" \
     #          f"max_spec={self.max_spec}\nmax_idx={self.max_idx}\nx_value={self.x_value}\nfunc={self.func})"

    def to_string(self):
        return f"Global_Function\nx_spec={self.x_spec}\ny_spec={self.y_spec}\n" \
               f"max_spec={self.max_spec}\nmax_idx={self.max_idx}\nx_value={self.x_value}\nfunc={self.func})"



    def split_at_max_value(self):
        left_gf = None
        right_gf = None

        if self.max_idx != 0:
            left_x_spec = self.x_spec[:self.max_idx]
            left_y_spec = self.y_spec[:self.max_idx]
            left_domain = self.domain <= self.x_value - self.p
            left_domain = self.domain[left_domain]
            left_gf = Global_Function(self.p,left_x_spec,left_y_spec,left_domain)
            
        if self.max_idx != self.x_spec.size-1:
            right_x_spec = self.x_spec[self.max_idx+1:]
            right_y_spec = self.y_spec[self.max_idx+1:]
            right_domain = self.domain > self.x_value + self.p 
            right_domain = self.domain[right_domain]
            right_gf = Global_Function(self.p,right_x_spec,right_y_spec,right_domain)
            
        return left_gf, right_gf


    def add_function(self, other):
        func = self.func
        other_func = other.func
        other_domain = other.domain
        
        if other_domain.size > 0 and self.domain.size > 0:
            step_size = self.domain[1] - self.domain[0]  # Assuming equidistant step size in domain
            indices = np.round((other_domain - self.domain[0]) / step_size).astype(int)
            func[indices] = other_func
        self.func = func





def set_global_function(gf):
    """
    This is a divide and conquer algorithm that generates a rectangular function of width 2*p over the largest amplitude 
    and calls itself both left and right of the rectangular function as long as there is left or right.
    """
    if gf is not None:
        func = Rectangle_Function(gf.p, gf.x_value, gf.max_spec, gf.domain)
        gf.func = func.rectangle
        left_gf, right_gf = gf.split_at_max_value()
        set_global_function(left_gf)
        set_global_function(right_gf)
        if left_gf is not None:
            gf.add_function(left_gf)
        if right_gf is not None:
            gf.add_function(right_gf)



def smooth_func(rectangle_func, p, length):
    """
    Smoothing a rectangle function with an appropriate mollifier. 
    Due to the convolution theorem the convolution itself can be computed by
    conv(f*g) = F^(-1)(F(f)*F(g))
    where F denotes the Fourier transform
    and F^(-1) its inverse
    """
    max_fun = np.amax(rectangle_func)
    fft_rectangle_func = np.fft.fft(rectangle_func)
    gaussian = scipy.signal.gaussian(M=length, std=p)
    # normalize
    gaussian /= sum(gaussian)
    fft_gaussian = np.fft.fft(gaussian)
    convolution = np.fft.ifft(fft_rectangle_func * fft_gaussian)
    convolution = np.roll(convolution,int(length/2))
    max_conv = np.amax(convolution)
    # scale the convolution such that its maximum equals the maximum of the spectrum
    convolution = convolution * max_fun / max_conv
    return convolution


def test_make_spectrum():
    domain = np.arange(0.,1.,0.1)
    x_spec = np.array([0.18243, 0.6123456])
    y_spec = np.array([3.4,8.])
    framerate = 3
    Spectrum = make_spectrum(x_spec,y_spec,domain,framerate)

test_make_spectrum()
  #### TEST

def test_make_y_spectrum():
    domain = np.arange(0.,1.,0.1)
    #print(domain)
    x_spec = np.array([0.18243, 0.6123456])
    y_spec = np.array([3.4,8.])
    #print(make_spectrum(x_spec,y_spec,domain))
    result = make_y_spectrum(x_spec,y_spec,domain)
    expected_result = np.array([0.,0.,3.4,0.,0.,0.,8.,0.,0.,0.])
    assert np.allclose(result, expected_result), "test_make_y_spectrum() do not match the expected values."

test_make_y_spectrum()

def test_Rectangle_Function():
    
    p = 2.
    x = 6.
    y = 4.
    domain = np.arange(11)
    rf = Rectangle_Function(p,x,y,domain)
    result = rf.rectangle
    #print(result)
    expected_result = np.array([0.,0.,0.,0.,0.,4.,4.,4.,4.,0.,0.])
    assert np.allclose(result, expected_result), "Rectangle function do not match the expected values."
    x = -1.
    rf = Rectangle_Function(p,x,y,domain)
    result = rf.rectangle
    expected_result = np.array([4.,4.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    #print(rf.rectangle)
    assert np.allclose(result, expected_result), "Rectangle function do not match the expected values."

test_Rectangle_Function()

def test_Rectangle_Funktion_1():
    p = 0.3
    x = 0.9
    y = 3.8
    domain = np.arange(0.,2.,0.1)
    #print(domain)
    rf = Rectangle_Function(p,x,y,domain)
    result = rf.rectangle
    #print(result)

test_Rectangle_Funktion_1()


def test_Global_Function():
    #print("test_Global_Function()")
    p = 2.
    x_spec = np.array([1.,4.,7.,9.])
    y_spec = np.array([1.,8.,7.,10.])
    domain = np.arange(11)
    gf = Global_Function(p,x_spec,y_spec,domain)
    set_global_function(gf)
    result = gf.func
    expected_result = np.array([1.,1.,1.,8.,8.,8.,8.,7.,10.,10.,10.])
    assert np.allclose(result, expected_result), "Global rectangle function do not match the expected values."
    p = 10.
    gf = Global_Function(p,x_spec,y_spec,domain)
    set_global_function(gf)
    result = gf.func
    expected_result = np.array([10.,10.,10.,10.,10.,10.,10.,10.,10.,10.,10.])
    assert np.allclose(result, expected_result), "Global rectangle function do not match the expected values."

test_Global_Function()




arr_1 = np.arange(0.1, 100., 0.25)
arr_2 = np.array([2., 6., 12.,13.])

indices = np.abs(arr_1 - arr_2[:, None]).argmin(axis=1)
##print(arr_1)
#print(indices)
point = 6.
index = np.abs(arr_1 - point).argmin()
#print(index)

#from scipy.fft import fft, fftfreq, fftshift
        
sig_1 = CosSignal(freq=10000, amp=2., offset=0)
sig_2 = SinSignal(freq=5000, amp=1., offset=0)
sig_3 = SinSignal(freq=3000, amp=3., offset=0)
sig = sig_1 + sig_2 + sig_3
wave = sig.make_wave(duration=10.5, start=0, framerate=44100)
wave.normalize()
y = wave.ys 
x = wave.ts
spectrum = wave.make_spectrum(full=True)
y = np.fft.fft(y)
#y = spectrum.hs
y = np.abs(y)
n =len(y)
d = 1 / wave.framerate
#y = spectrum.hs
x = np.fft.fftfreq(n,d)
x = spectrum.fs



ig, ax = plt.subplots()
ax.plot(x,y)
#plt.show()

