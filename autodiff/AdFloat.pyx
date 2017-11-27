# from math import log
import numpy as np
cimport numpy as np
ctypedef np.float_t DTYPE_t
cdef class AdFloat:

    cdef public double x
    cdef public double dx
    def __init__(self, x, dx=1):
        self.x = x
        self.dx = dx

    def __add__(self, b):
        """
        d/dx(a+b) = d/dx(a) + d/dx(b)
        """
        a = self
        if not isinstance(a, AdFloat):
            a = AdFloat(a, 0)
        if not isinstance(b, AdFloat):
            b = AdFloat(b, 0)
        return AdFloat(a.x+b.x, a.dx + b.dx)
        
    

    def __mul__(self, b):
        """
        d/dx(a*b) = d/dx(a)*b + d/dx(b)*a
        """
        a = self
        if not isinstance(a, AdFloat):
            a = AdFloat(a, 0)
        if not isinstance(b, AdFloat):
            b = AdFloat(b, 0)

        return AdFloat(a.x*b.x, a.dx*b.x + b.dx*b.x)



    def __sub__(self, b):
        """
        d/dx(a-b) = dx(a) - d/dx(b)
        """
        a = self
        if not isinstance(a, AdFloat):
            a = AdFloat(a, 0)
        if not isinstance(b, AdFloat):
            b = AdFloat(b, 0)
        
        return AdFloat(a.x-b.x, a.dx - b.dx)

    #Legacy stuff for python 2
    def __div__(self, b):
        """
        d/dx(a/b) = ( d/dx(a)*b- d/dx(b)*a ) / (b^2)
        """
        a = self
        if not isinstance(a, AdFloat):
            a = AdFloat(a, 0)
        if not isinstance(b, AdFloat):
            b = AdFloat(b, 0)
        
        return AdFloat(a.x/b.x, (a.dx*b.x - b.dx*a.x)/b.x**2)

    #New division operation from python 3
    def __truediv__(self, b):
        """
        d/dx(a/b) = ( d/dx(a)*b- d/dx(b)*a ) / (b^2)
        """
        a = self
        if not isinstance(a, AdFloat):
            a = AdFloat(a, 0)
        if not isinstance(b, AdFloat):
            b = AdFloat(b, 0)
        
        return AdFloat(a.x/b.x, (a.dx*b.x - b.dx*a.x)/b.x**2)

    
    def __pow__(self, b, z):
        """
        d/dx(a**b) = b*a^(b-1)*d/dx(a) + ln(a)*a^b*d/dx(b)
        """
        a = self
        if not isinstance(a, AdFloat):
            a = AdFloat(a, 0)
        if not isinstance(b, AdFloat):
            b = AdFloat(b, 0)
        #Both a and b are variables
        if a.dx == 1 and b.dx == 1:
            #avoid log(0)
            if a.x==0:
                return AdFloat(a.x**b.x, b.x*a.x**(b.x-1)*a.dx)

            return AdFloat(a.x**b.x, b.x*a.x**(b.x-1)*a.dx + log(a.x)*a.x**b.x*b.dx)
        
        # a is a variable, and b in a constant
        elif a.dx==1:
            
            return AdFloat(a.x**b.x, b.x*a.x**(b.x-1)*a.dx)
        
        # b is a variable, and a is a constant
        elif b.dx==1:
            #avoid log(0)
            if a.x==0:
                return AdFloat(a.x**b.x, 0)
            return AdFloat(a.x**b.x, a.x**b.x*log(a.x)*b.dx)
        
        return AdFloat(a.x**b.x, b.x*a.x**(b.x-1)*a.dx)

    
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rtruediv__ = __truediv__
    __rpow__ = __pow__

def sqrt(a):
    if isinstance(a, AdFloat):
        return AdFloat(np.sqrt(a.x), a.dx/(2*np.sqrt(a.x)))
    return np.sqrt(a)
cpdef sin(a):
    if isinstance(a, AdFloat):
        return AdFloat(np.sin(a.x), np.cos(a.x)*a.dx)
    return np.sin(a)
cpdef cos(a):
    if isinstance(a, AdFloat):
        return AdFloat(np.cos(a.x), -np.sin(a.x)*a.dx)
    return np.cos(a)
cpdef tan(a):
    if isinstance(a, AdFloat):
        return AdFloat(np.tan(a.x), a.dx/np.cos(a.x)**2)
    return np.tan(a)
# cpdef erf(a):
#     if isinstance(a, AdFloat):
#         return AdFloat(errorfunc(a.x), 2.0 / np.sqrt(np.pi) * np.exp(-(a.x ** 2.0)) * a.dx)
#     return errorfunc(a)
cpdef exp(a):
    if isinstance(a, AdFloat):
        return AdFloat(np.exp(a.x), np.exp(a.x)*a.dx)
    return np.exp(a)
cpdef log(a):
    if isinstance(a, AdFloat):
        return AdFloat(np.log(a.x), a.dx/a.x)
    return np.log(a)

cpdef partial(f, x, i):
    """
    Computes the partial derivative of f(x) with respect to x_i
    This is done by setting d/dx_j (x)=0 forall j ≠ i
    """
    result = f(*[AdFloat(x_j, j == i) for j, x_j in enumerate(x)])
    return result.dx


cpdef jacobian(f, np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] param):
    """
    Calculates the jacobian matrix for d/dx_i (f(x))
    """

    cdef int N = len(x)
    cdef int M = len(param)
    cdef np.ndarray[double, ndim=2] J = np.zeros((N, M))
    cdef int i,j,k
    cdef double x_i, p_k
    for i, x_i in enumerate(x):
        for j in range(M):
            parameters=[]
            for k, p_k in enumerate(param):
                    parameters.append(AdFloat(p_k, k == j))
            val = f(AdFloat(x_i,0), parameters)
            J[i, j] = val.dx

    return J
