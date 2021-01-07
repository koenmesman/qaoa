from math import pi
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
plt.interactive(True)
backend = Aer.get_backend('qasm_simulator')
from scipy.optimize import minimize
import nlopt
from numpy import *


qaoa_shots = 10000


def eval_cost(out_state, graph):
    # evaluate Max-Cut
    v = graph[0]
    edges = graph[1]
    c = 0
    bin_len = "{0:0" + str(v) + "b}"  # string required for binary formatting
    bin_val = list(bin_len.format(list(out_state).index(max(out_state))))
    bin_val = [int(i) for i in bin_val]
    bin_val = [-1 if x == 0 else 1 for x in bin_val]

    for e in edges:
        c += 0.5 * (1 - int(bin_val[e[0]]) * int(bin_val[e[1]]))
    return c

# QAOA function
def qaoa_circ(params, graph, p):
    beta, gamma = params
    v, edge_list = graph
    vertice_list = list(range(0, v, 1))
    if beta < 0 or beta > (2*pi) or gamma < 0 or gamma > (2*pi):
        return 0
    else:
        qc = QuantumCircuit(v, v)
        for qubit in range(v):
            qc.h(qubit)
        for iteration in range(p):
            for e in edge_list:                     # TODO: fix for unordered edges e.g. (2,0)
                qc.cnot(e[0], e[1])
                qc.rz(-gamma, e[1])
                qc.cnot(e[0], e[1])
            for qb in vertice_list:
                qc.rx(2*beta, qb)
        qc.measure(range(v), range(v))
        result = execute(qc, backend, shots=qaoa_shots).result()
        out_state = result.get_counts()
        #print(qc)

    prob = list(out_state.values())
    states = list(out_state.keys())
    exp = 0
    for k in range(len(states)):
        exp += eval_cost(states[k], graph)*prob[k]/qaoa_shots
    return exp


def qaoa_landscape(g, n):
    betas = np.linspace(0, 2 * pi, n)
    gammas = np.linspace(0, 2 * pi, n)
    val = [[0]*n for _ in range(n)]
    for b in range(n):
        beta = betas[b]
        for c in range(n):
            gamma = gammas[c]
            val[b][c] = qaoa_circ([beta, gamma], g)

    X, Y = np.meshgrid(betas, gammas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, np.array(val), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()
    return