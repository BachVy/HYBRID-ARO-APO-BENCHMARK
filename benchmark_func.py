import numpy as np

# Hàm Sphere
def sphere(x):
    return np.sum(x**2)

# Hàm Schwefel 2.22
def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# Hàm Schwefel 2.21
def schwefel_2_21(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += np.sum(x[0:i+1])**2
    return o

# Hàm Max Absolute
def max_absolute(x):
    return np.max(np.abs(x))

# Hàm Generalized Power
def generalized_power(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += np.abs(x[i])**(i + 1)
    return o

# Hàm Weighted Sphere
def weighted_sphere(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += (i + 1) * x[i]**2  
    return o

# Hàm Composite Quadratic
def composite_quadratic(x):
    D = len(x)
    return np.sum(x**2) + np.sum(0.5 * D * (x**2)) + np.sum(0.5 * D * (x**4))

# Hàm Ackley
def ackley(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)

# Hàm Schwefel 2.6 Simple
def schwefel_2_6_simple(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))