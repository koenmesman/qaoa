import numpy as np
from math import pi
from qiskit import (
  QuantumCircuit,
  execute,
  Aer)
import time
import networkx as nx
import random
from matplotlib import cm
import matplotlib.pyplot as plt
plt.interactive(True)
backend = Aer.get_backend('qasm_simulator')
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


# random graph generator
def generate_graph(n, r):
    nodes = list(range(0, n))
    e = int(round(n*r))
    edges = [[0]*2 for _ in range(e)]
    super_flag = 1
    timeout = time.time() + 60 * 3
    while super_flag:
        for x in range(e):
            flag = 1
            while flag:
                node_a = nodes
                edges[x][0] = random.choice(node_a[:(n-1)])
                node_b = nodes
                edges[x][1] = random.choice(node_b[edges[x][0] + 1:])
                if edges.count(edges[x]) > 1:
                    flag = 1
                else:
                    flag = 0
        super_flag = 0
        for a in nodes:
            if (sum(x.count(a) for x in edges)) == 0:
                super_flag = 1
            else:
                if time.time() > timeout:
                    print("Timeout error: Not all nodes could be connected in expected time. ",
                          "Please increase edge to vertice ratio or increase timeout value")
                    break
    return [n, edges]


def eval_cost(out_state):
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


def find_exp(graph, out_state):
    #evaluate the expectation value
    prob = list(out_state.values())
    states = list(out_state.keys())
    exp = 0
    for k in range(len(states)):
        exp += eval_cost(states[k])*prob[k]/qaoa_shots
    return exp


def qaoa_inv(params):
    states = qaoa_circ(params)
    best = max(states, key=states.get)
    out = eval_cost(best)
    if out == 0:
        inv = 2
    else:
        inv = 1/out
    return inv


# QAOA function
def qaoa_circ(params):
    beta, gamma = params
    v, edge_list = graph

    if beta < 0 or beta > (2*pi) or gamma < 0 or gamma > (2*pi):
        return 10
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

    return out_state


def qaoa_landscape(g, n):

    betas = np.linspace(0, 2 * pi, n)
    gammas = np.linspace(0, 2 * pi, n)
    val = [[0]*n for _ in range(n)]
    for b in range(n):
        beta = betas[b]
        for c in range(n):
            gamma = gammas[c]
            val[b][c] = find_exp(g, qaoa_circ([beta, gamma], g))

            #print(val[b][c])
    X, Y = np.meshgrid(betas, gammas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, np.array(val), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()
    return


######################################################################################
nr_nodes = 3
ratio = 1
p = 1
qaoa_shots = 10000

vertice_list = list(range(0, nr_nodes, 1))

#generate a (pseudo) random graph
graph = generate_graph(nr_nodes, ratio)
edge_list = graph[1]

G = nx.Graph()
G.add_nodes_from(vertice_list)
G.add_edges_from(edge_list)

# Generate plot of the Graph
colors = ['r' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos = nx.spring_layout(G)
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

#############################################################
#qaoa_landscape(graph, 40)  # nr_nodes, edge list, resolution


init_param = [pi, pi]
cuts = minimize(qaoa_inv, init_param, method='nelder-mead',
               options={'xatol': 1e-4, 'disp': True}).fun

print("cuts = ", 1/cuts)
G.clear()


# TODO
#   fix qaoa func
#       find best expectation value
#       evaluate expectation?
#   try different optimizers
#   readout success rate and runtime
#       finding maximum success rate/accuracy
#       Find runtime for set accuracy
#   scale results
#   plot?
