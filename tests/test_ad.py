from autodiff import AdFloat, jacobian
import numpy as np
def test_initialization():
    a = AdFloat(1)
    b = AdFloat(1,1)
    assert a.x == b.x
    assert a.dx == b.dx

def test_jacobian():
    import numpy as np
    x = np.arange(0,10, dtype=np.float64).reshape(10,1)
    p = np.ndarray([1,2], dtype=np.float64).reshape(2,1)
    
    def f(x,p):
        return p[0]*x + p[1]
    
    a = jacobian(f,x,p) 
    b = np.array([[0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1]])
    assert (a==b).all()

def test_sin():
    from autodiff import sin
    x = AdFloat(np.pi)
    f = lambda x: sin(x)*2

    assert f(x).dx == np.cos(np.pi)*2
    assert f(x).x  == np.sin(np.pi)*2

def test_cos():
    from autodiff import cos
    x = AdFloat(np.pi)
    f = lambda x: cos(x)*2

    assert f(x).dx == -np.sin(np.pi)*2
    assert f(x).x  == np.cos(np.pi)*2

def test_tan():
    from autodiff import tan
    x = AdFloat(np.pi)
    f = lambda x: tan(x)*2

    assert f(x).dx == 1/(np.cos(np.pi)**2)*2
    assert f(x).x  == np.tan(np.pi)*2
    
def test_operations():
    x = AdFloat(1)
    #addition
    1.0+x
    x+1.0
    #subtraction
    1.0-x
    x-1.0
    #multiplication
    1.0*x
    x*1.0
    #division
    1.0/x
    x/1.0
    #power
    1.0**x
    x**1.0