import numpy as np
import math
from scipy.special import gamma
from benchmark_func import sphere, schwefel_2_22, schwefel_2_21, max_absolute, generalized_power, weighted_sphere, composite_quadratic, ackley, schwefel_2_6_simple

# Cấu hình chung
N = 50          # Số cá thể
T = 1000        # Số vòng lặp
dim = 70        # Số chiều
# benchmark_names = ["sphere", "schwefel_2_22", "schwefel_2_21", "max_absolute", "generalized_power", "weighted_sphere", "composite_quadratic", "ackley", "schwefel_2_6_simple"]
benchmark_names = ["sphere", "schwefel_2_22", "max_absolute", "generalized_power", "weighted_sphere", "composite_quadratic"]
# Các hàm benchmark
def benchmark_functions(x, fun_index):
    # functions = [sphere, schwefel_2_22, schwefel_2_21, max_absolute, generalized_power, weighted_sphere, composite_quadratic, ackley, schwefel_2_6_simple]
    functions = [sphere, schwefel_2_22, max_absolute, generalized_power, weighted_sphere, composite_quadratic]
    
    if 0 <= fun_index < len(functions):
        return functions[fun_index](x)
    else:
        raise ValueError("fun_index phải từ 0 đến 8")

# Hàm thiết lập biên
def set_bounds(fun_index, dim):
    pi = math.pi
    bounds = {
        0: (-5, 5),      # sphere
        1: (-10, 10),    # schwefel_2_22
        2: (-100, 100),  # schwefel_2_21
        3: (-100, 100),  # max_absolute
        4: (-2, 2),      # generalized_power
        5: (-100, 100),  # weighted_sphere
        6: (-100, 100),  # composite_quadratic
        7: (-32, 32),    # ackley
        8: (-100, 100),  # schwefel_2_6_simple
    }
    lb, ub = bounds[fun_index]
    return np.full(dim, lb), np.full(dim, ub)

# Hàm hỗ trợ
def Levy(dim):
    beta = 1.5
    sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) / (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / np.abs(v)**(1/beta)

def SpaceBound(position, ub, lb):
    return np.clip(position, lb, ub)