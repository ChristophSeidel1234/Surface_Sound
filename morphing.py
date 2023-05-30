import numpy as np 
import math


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
            first_idx = self.domain[0]
            shifted_other_domain = other_domain -first_idx
            func[shifted_other_domain] = other_func
        self.func = func



def set_global_function(gf):
    """
    This is a recursive method that actually has the structure of a binary tree and sets the global rectangle function to the object gf.
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

                

  #### TEST

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


def test_Global_Function():
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


    
        
