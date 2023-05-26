import numpy as np 
import math



def rectangle(t,x_spec,y_spec,p):
    if t <= x_spec - p or t >= x_spec + p:
        return 0.0
    else:
        return y_spec

class Rectangle_Function:


    def __init__(self, p, x, y,domain):
        self.p = p
        self.x = x
        self.y = y
        self.domain = domain
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
            print(' ')
            print('left')
            left_x_spec = self.x_spec[:self.max_idx]
            #print("left_x_spec " + str(left_x_spec))
            left_y_spec = self.y_spec[:self.max_idx]
            left_domain = self.domain <= self.x_value - self.p
            
            left_domain = self.domain[left_domain]
            left_gf = Global_Function(self.p,left_x_spec,left_y_spec,left_domain)
            print(left_gf.to_string())
        if self.max_idx != self.x_spec.size-1:
            print(' ')
            print('right')
            #print(self.to_string())
            #print('self.max_idx = '+str(self.max_idx))
            #print('self.x_spec.size-1 = '+str(self.x_spec.size))
            #print("parent spec " + str(x_spec))
            right_x_spec = self.x_spec[self.max_idx+1:]
            #print("right_x_spec " + str(right_x_spec))
            right_y_spec = self.y_spec[self.max_idx+1:]
            right_domain = self.domain > self.x_value + self.p
            right_domain = self.domain[right_domain]
            right_gf = Global_Function(self.p,right_x_spec,right_y_spec,right_domain)
            print(right_gf.to_string())
            
        return left_gf, right_gf


    def add_function(self, other):
        func = self.func
        other_func = other.func
        other_domain = other.domain
        func[other_domain] = other_func
        self.func = func






def set_global_function(gf):
    if gf is not None:
        
        func = Rectangle_Function(gf.p, gf.x_value, gf.max_spec, gf.domain)
        print(func.to_string())
        gf.func = func.rectangle
        left_gf, right_gf = gf.split_at_max_value()
        

        set_global_function(left_gf)
        set_global_function(right_gf)
        if left_gf is not None:
            gf.add_function(left_gf)
        if right_gf is not None:
            gf.add_function(right_gf)

        #return func
                


  #### TEST
p = 10.
x_spec = np.array([9.,12.,50.,66.])
y_spec = np.array([10.,34.,50.,30.])
domain = np.arange(99)

gf = Global_Function(p, x_spec, y_spec, domain)
#print(gf.to_string())
left_gf, right_gf = gf.split_at_max_value()
#print(left_gf.to_string())
#print(right_gf.to_string())
count = 11
set_global_function(gf)
print(" ")
print(gf.to_string())

    
        
