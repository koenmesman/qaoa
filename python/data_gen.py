import numpy as np
import csv
from math import pi
from scipy.optimize import curve_fit
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
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import optimize



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


def find_exp(out_state, graph):
    #evaluate the expectation value
    prob = list(out_state.values())
    states = list(out_state.keys())
    exp = 0
    for k in range(len(states)):
        exp += eval_cost(states[k], graph)*prob[k]/qaoa_shots
    return exp


def qaoa_inv(params, graph):
    states = qaoa_circ(params, graph)
    #best = max(states, key=states.get)
    #out = eval_cost(best)
    out = find_exp(states, graph)
    if out == 0:
        inv = 2
    else:
        inv = 1/out
    return inv


def qaoa_norm(params, v, e):
    graph = [v, e]  # graph for some reason doesn't pass normally
    states = qaoa_circ(params, graph)
    out = find_exp(states, graph)
    return out


# QAOA function
def qaoa_circ(params, graph):
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


def line(x, a, b):
    return a * x + b


def rand_BFGS(graph):
    v, edge_list = graph
    best = 2
    for i in range(2*v):
        param = np.random.uniform(low=0.0, high=2*pi, size=2).tolist()
        tmp = minimize(qaoa_inv, param, args=graph, method='BFGS',
                 options={'disp': False})
        if tmp < best:
            best = tmp
    return best


def switch_opt(x, init_param, graph):
    bounds = [(0, 2 * pi), (0, 2 * pi)]
    v, e = graph
    return {
        'nm': minimize(qaoa_inv, init_param, args=graph, method='nelder-mead',
                              options={'ftol': 1e-2, 'maxfev': 400, 'disp': False}),
        'BFGS': minimize(qaoa_inv, init_param, args=graph, method='BFGS',
                              options={'disp': False}),
        'shgo': optimize.shgo(qaoa_norm, bounds, args=(v, e))
        #'rBFGS': rand_BFGS(graph)
    }[x]


def c_opt(opt_method, data_path, problem_range):
    d_time = []
    data_points = []
    data_x = []
    for problem_size in problem_range:  # loop over problem size

        # generate a (pseudo) random graph
        graph = generate_graph(problem_size, ratio)
        edge_list = graph[1]

        for d in range(data_shots):  # loop over data points per problem size
            init_param = [pi, pi]
            start = time.time()

            result = switch_opt(opt_method, init_param, graph)

            d_time.append(time.time() - start)
            print("now finished size ", problem_size, " number ", d)
            data_x.append(problem_size)

    try:
        load_data = np.loadtxt('Test_data/' + data_path)
        save_data = [data_x, d_time]
        data = np.hstack((load_data, save_data))
    except IOError:
        data = [data_x, d_time]
    finally:
        np.savetxt('Test_data/' + data_path, data)
        print("data saved for method " + opt_method + " for problem size ", problem_size)


######################################################################################
ratio = 1
p = 1
qaoa_shots = 10000
data_shots = 3



#qaoa_landscape(graph, 40)  # nr_nodes, edge list, resolution

for iterator in range(1, 6):
    c_opt("shgo", "nm_shgo.dat", range(iterator*3, (iterator+1)*3))
    c_opt("nm", "nm_data.dat", range(iterator*3, (iterator+1)*3))
    c_opt("BFGS", "BFGS_data.dat", range(iterator*3, (iterator+1)*3))
#    c_opt("rBFGS", "rBFGS_data.dat", range(iterator*3, (iterator+1)*3))
