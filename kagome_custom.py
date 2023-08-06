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

from qiskit.primitives import Estimator


class custom_VQE():
    
    def __init__(self, edge_list, num_qubits, layers, shots):
        self.edge_list = edge_list
        self.num_qubits = num_qubits
        self.layers = layers
        self.shots = shots
        self.down_list = [11,10,2]
        self.num_param = 3*self.layers
        
        self.get_backend()
        self.setup_circuit()
        self.add_init_state()
        self.build_ansatz()
        self.add_ansatz()
        self.build_measure_circuits()
        
        graph = rx.PyGraph(multigraph=False)
        graph.add_nodes_from(range(self.num_qubits))
        graph.add_edges_from(self.edge_list)

        self.kagome_unit_cell = Lattice(graph)
        
        self.estimator = Estimator()
        heis = HeisenbergModel.uniform_parameters(lattice=self.kagome_unit_cell,uniform_interaction=1.0,uniform_onsite_potential=0.0)
        log_mapper = LogarithmicMapper()
        self.ham = 4*log_mapper.map(heis.second_q_ops().simplify())

        
       
    
    def plot_lattice(self):
        kagome_pos = {0:[0,6.8], 6:[0.6,5], 9:[1.8,5.1], 
                      1:[0,-0.8], 2:[-0.6,1], 4:[0.6,1], 10:[1.2,3], 
                      11:[-0.6,5], 5:[-1.2,3], 3:[-1.8,0.9], 
                      8:[-1.8,5.1], 7:[1.8,0.9]}
        self.kagome_unit_cell.draw(style={'with_labels':True, 'font_color':'white', 'node_color':'purple', 'pos':kagome_pos})
        plt.show()
        
    
    def exact_gs_energy(self):
        exact_solver = NumPyEigensolver(k=3)
        exact_result = exact_solver.compute_eigenvalues(self.ham)

        self.gs_energy = np.round(exact_result.eigenvalues[0], 4)
        print('Exact ground state energ: ' + str(self.gs_energy))
        
        
    
    
    # ----------------------- Base quantum circuit setup ---------------------------
    def setup_circuit(self):
        self.qc = QuantumCircuit(self.num_qubits, self.num_qubits)
    
    def add_init_state(self):
        for i in self.down_list:
            self.qc.x(i)

#     def add_init_state(self):
        
#         for i in range(self.num_qubits):
        
#             if i in self.down_list:
#                 self.qc.x(i)
#                 self.qc.h(i)
#                 self.qc.t(i)
#             else:
#                 self.qc.h(i)
#                 self.qc.t(i)
         
            
    def add_edge_gates(self,qa,edge,var,p):
        [n1, n2] = [edge[0],edge[1]]
        
        phi = self.param[var+3*p]
     
        if var == 0:
            qa.rxx(phi,n1,n2)
        elif var == 1:
            qa.ryy(phi,n1,n2)
        else :
            qa.rzz(phi,n1,n2)

    def build_ansatz(self):
        self.param = []
        for k in range(self.num_param):
            self.param.append(Parameter('θ_' + str(k)))

        self.ansatz = QuantumCircuit(self.num_qubits,name='ansatz')
        
        self.ansatz.barrier()
        for k in range(self.layers):
            for i in range(3):
                for edge in self.edge_list:
                    self.ansatz.barrier()
                    self.add_edge_gates(self.ansatz,edge,i,k)
        self.ansatz.barrier()
        
    def add_ansatz(self):
        self.qc.append(self.ansatz, range(self.num_qubits))

  
            
    # ---------------- Measurement and energy evaluation ----------------- 
    def get_backend(self):
        self.aer_sim = Aer.get_backend('aer_simulator')
    
    def build_measure_circuits(self):
        self.qc_meas = [None, None, None]
        for j in range(3):
            self.qc_meas[j] = self.qc.copy()
            for i in range(self.num_qubits):
                if j == 0:
                    self.qc_meas[j].h(i)

                elif j == 1:
                    self.qc_meas[j].s(i)
                    self.qc_meas[j].h(i)
                    self.qc_meas[j].x(i)
                    
                    

                self.qc_meas[j].measure(i,i)
                
                





    

    
        
    
        
  
        
            