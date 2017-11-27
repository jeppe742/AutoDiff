# AutoDiff - A framework for doing automatical differentiation

## Basic Usage
The framework works around the `AdFloat` object. To evaluate a function for a value, you need to create this value as an `AdFloat`

```python
from autodiff import AdFloat
x = AdFloat(27)
```
simple operations like + - * / is overloaded, so you can just do

```python
f = lambda x: x*2 
value = f(x)
```
`value` is now an `AdFloat`, which means that you can access both the derivative, and the evaluated value by

```python
print(value.dx)
> 2
print(value.x)
> 52
```
For more complex functions, you have to import the implemented versions.

```python
from autodiff import AdFloat, sin
x = AdFloat(3.14159265)
f = lambda x: sin(x)
value = f(x)
print(value.dx)
> -1
```
## Chaining derivatives
The power of automatic differentiation comes from the fact, that one can evaluate derivatives of arbitrary functions without nummeric approximations.  
To work with functions with multiple terms, simply write them as normal

```python
from autodiff import AdFloat, sin, exp
x = AdFloat(1)
f = lambda x: sin(exp(x*4)) + (x+2)**2
print(f(x).dx)
> -74.94977972639266 
```

## Installation
### Windows
![WINDOWS](https://ci.appveyor.com/api/projects/status/cfl5wo6adujm7bac?svg=true)  
AutoDiff is available as a precompiled wheel for windows, and can be installed using

`pip install autodiff`

### Other platforms
For other platforms it can be build from source by downloading the repository, and running  
`python setup.py install`  

This requires that you have installed a c compiler
