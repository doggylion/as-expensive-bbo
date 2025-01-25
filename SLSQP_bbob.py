from scipy.optimize import minimize
import numpy


import random

import benchmark.bbobbenchmarks as bn
import numpy as np

#　目的関数
# def objective_fnc(x):
#     x1 = x[0]
#     x2 = x[1]
#     return x1**2 + x1*x2 

# #　等式制約条件  
# def equality_constraint(x):
#     x1 = x[0]
#     x2 = x[1]
#     return x1**3 + x1*x2 - 100

# # 不等式制約条件
# def inequality_constraint(x):
#     x1 = x[0]
#     x2 = x[1]
#     return x1**2 + 4*x2 - 50


# SEED = int(sys.argv[1])
SEED = int(1)

random.seed(SEED)

dim = 20
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F21()

bounds = (lb,ub)
bound = [bounds] * dim

print(bound)

def fitness(x):
    if type(x) is np.ndarray:
        x = x.tolist()
    return OBJECTIVE_FUNCTION(np.array(x)) - OBJECTIVE_FUNCTION.fopt


x0=[random.uniform(-5,5)] * dim

print(x0)

result=minimize(fitness, x0, method="SLSQP", bounds=bound)

print(result)