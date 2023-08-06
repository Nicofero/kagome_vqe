import numpy as np
from time import time
import matplotlib.pyplot as plt
import rustworkx as rx

from qiskit_nature.problems.second_quantization.lattice import Lattice
import sys
import os
import time

from heisenberg_model import HeisenbergModel
from qiskit_nature.mappers.second_quantization import LogarithmicMapper
from qiskit.algorithms import NumPyEigensolver
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram

import dask
from distributed import Client
import dask.array as da

from kagome_custom import custom_VQE

from qiskit.primitives import Estimator
from qiskit.algorithms.optimizers import SPSA


from scipy.optimize import differential_evolution
import toolz
from multiprocessing import Pool
from qiskit.algorithms.optimizers import SLSQP, L_BFGS_B


t = 1.0
edge_list = [
    (1, 2, t),
    (2, 3, t),
    (3, 5, t),
    (5, 8, t),
    (8, 11, t),
    (11, 0, t),
    (0, 6, t),
    (6, 9, t),
    (9, 10, t),
    (10, 7, t),
    (7, 4, t),
    (4, 1, t),
    (4, 2, t),
    (2, 5, t),
    (5, 11, t),
    (11, 6, t),
    (6, 10, t),
    (10, 4, t),
]


num_qubits = 12
layers = 6
shots = 100



kagome = custom_VQE(edge_list, num_qubits, layers, shots)
kagome.get_backend()


file_name = 'remix_dataREAL8.txt'

def objective(x):
    global kagome
    job = kagome.estimator.run([kagome.qc],[kagome.ham],[x])
    energy = job.result().values[0]
    #print(energy)
    return energy

def callback(xk,convergence):
    global file_name
    ek = objective(xk)
    #print(ek)
    file = open(file_name,'a')
    ek = objective(xk)
    file.write(str(ek))
    file.write('\n')
    file.close()


file = open(file_name,'w')
file.write(str(objective([0]*18)))
file.write('\n')
file.close()



#-------------DE-----------------
initial_point0 = np.random.rand(18)*1.e-2
bounds = [(-np.pi,np.pi)]*18
maxiter=100000
popsize=2
tol=1.e-10
pool = Pool(2*18)
parameters={'maxiter':maxiter,
           'popsize':popsize,
           'workers':pool.map,
           'polish':False,
           'strategy':'best1exp',
           'updating':'deferred'}



optimizerDE = differential_evolution(func=objective,bounds=bounds,x0=initial_point0,callback=callback,disp=True,tol=tol,**parameters)

#---------------SLSQP-------------------------------
# optimizerSLSQP = SLSQP(maxiter=500,disp=True)
# initial_point1 = optimizerDE.x
# values1=optimizerSLSQP.minimize(fun=objective,x0=initial_point1)
# file = open('remix_data1.txt','a')
# file.write(str(values1.fun))
# file.write('\n')
# file.close()


#---------------L-BFGS-B-------------------------------
optimizerL_BFGS_B = L_BFGS_B(maxfun=1000*18,maxiter=10000,ftol=1.e-18)
initial_point1 = optimizerDE.x
values1=optimizerL_BFGS_B.minimize(fun=objective,x0=initial_point1)
file = open(file_name,'a')
file.write(str(values1.fun))
file.write('\n')
file.close()


    










    

    
