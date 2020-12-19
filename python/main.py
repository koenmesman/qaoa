import numpy as np
from math import pi
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
import networkx as nx
import matplotlib.pyplot as plt
plt.interactive(True)

from scipy.optimize import minimize

v = 5
vertice_list = list(range(0, v, 1))
G = nx.Graph()
G.add_nodes_from(vertice_list)

# tuple is (i, j) where (i, j) is the edge
# This line just tells our graph how the edges are connected to each other
edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (2, 4), (3, 4)]
G.add_edges_from(edge_list)

# Generate plot of the Graph
colors       = ['r' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos = nx.spring_layout(G)
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

#############################################################
p = 2
n = 5      # grid size
betas = np.linspace(0, 2*pi, n)
gammas = np.linspace(0, 2*pi, n)
backend = Aer.get_backend('statevector_simulator')
bin_len = "{0:0" + str(v) + "b}"

bin_val = 0
max_cost = 0
cuts = 0
b_opt = 0
c_opt = 0

for b in betas:
    for c in gammas:
        qc = QuantumCircuit(v, v)
        # print('beta = ', b, 'gamma = ', c)
        for qubit in range(v):
            qc.h(qubit)
        for iteration in range(p):
            for e in edge_list:                     # TODO: fix for unordered edges e.g. (2,0)
                qc.cnot(e[0], e[1])
                qc.rz(c, e[1])
                qc.cnot(e[0], e[1])
            for qb in vertice_list:
                qc.rx(b, qb)
                qc.measure(range(v), range(v))
                result = execute(qc, backend).result()
                out_state = result.get_statevector()
                # plot_histogram(result.get_counts())
                # print(out_state)
                # print(qc)

                # find binary value of most probable state
                bin_val = list(bin_len.format(list(out_state).index(max(out_state))))
                # tst = list(out_state).index(max(out_state))
                bin_val = [int(i) for i in bin_val]
                bin_val = [-1 if x == 0 else 1 for x in bin_val]
                # print(bin_val)

                cost_sum = 0
                for e in edge_list:
                    cost_sum += 0.5 * (-int(bin_val[e[0]]) * int(bin_val[e[1]]) + 1)
                if cost_sum > max_cost:
                    max_cost = cost_sum
                    cuts = bin_val
                    b_opt = b
                    c_opt = c
                # print(out_state)
print(max_cost)
print(cuts)
print('beta = ', b_opt, 'gamma = ', c_opt)
G.clear()
