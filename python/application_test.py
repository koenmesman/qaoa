from math import pi, acos, sqrt
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from qaoa_def import cost_tsp
from math import pi, acos
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator

plt.interactive(True)
#backend = Aer.get_backend('qasm_simulator')
from qiskit.circuit.library import RYGate, RZZGate, OR, ZGate
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
import nlopt
import numpy as np
import qiskit.converters.circuit_to_gate
from numpy import *
from Classic_opt import shgo_fun


plt.interactive(True)
from scipy.optimize import minimize
import nlopt
import numpy as np
import Classic_opt as co
from qaoa_def import qaoa_landscape, tsp_circ, dicke_init, dsp_cost
from data_gen import generate_graph, test_graph, tsp_graph
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
backend = Aer.get_backend('qasm_simulator')



def write_data(save_data, data_name):
    data_path = 'Test_data/' + data_name + 'timing_data.dat'
    try:
        load_data = np.loadtxt(data_path)
        save_data = np.array(save_data)
        print(save_data)
        data = np.hstack((load_data, save_data))
    except IOError:
        data =  np.array(save_data)
    finally:
        np.savetxt(data_path, data)
        print("data saved for " + data_name)


def run_qaoa(q_app, opt_func, graph, p):
    init_param = [pi, pi]
    result = opt_func(init_param, graph, p, q_app)
    return result

def data_test(q_app, problem_range, data_shots, opt_func, p):
    d_time = []
    data_x = []
    for problem_size in problem_range:  # loop over problem size
        ratio = 1
        # generate a (pseudo) random graph
        graph = {
            'max-cut': generate_graph(problem_size, ratio),
            'tsp': tsp_graph(problem_size, ratio),
            'dsp': generate_graph(problem_size, ratio)
        }.get(q_app)

        for d in range(data_shots):  # loop over data points per problem size, time optimizer
            start = time.time()
            run_qaoa(q_app, opt_func, graph, p)

            d_time.append(time.time() - start)
            print("now finished size ", problem_size, " number ", d)
            data_x.append(problem_size)
            data = [data_x, d_time]
            write_data(data, q_app)

#data_test('tsp', range(3, 6), 20, shgo_fun, 1)
data_test('dsp', range(11, 12), 3, shgo_fun, 1)

