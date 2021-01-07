from math import pi
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
plt.interactive(True)
backend = Aer.get_backend('qasm_simulator')
from scipy.optimize import minimize
import nlopt
import numpy as np
import Classic_opt as co


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


def c_opt(opt_func, opt_type, problem_range):
    d_time = []
    data_x = []
    for problem_size in problem_range:  # loop over problem size

        # generate a (pseudo) random graph
        graph = generate_graph(problem_size, ratio)

        for d in range(data_shots):  # loop over data points per problem size, time optimizer
            init_param = [pi, pi]
            start = time.time()

            result = opt_func(init_param, graph, p)

            d_time.append(time.time() - start)
            print("now finished size ", problem_size, " number ", d)
            data_x.append(problem_size)

    # add results data if file exist, otherwise create new file from data
    data_path = 'Test_data/' + opt_type + '_data.dat'
    try:
        load_data = np.loadtxt(data_path)
        save_data = [data_x, d_time]
        data = np.hstack((load_data, save_data))
    except IOError:
        data = [data_x, d_time]
    finally:
        np.savetxt(data_path, data)
        print("data saved for method " + opt_type + " for problem size ", problem_size)

    return result


######################################################################################
ratio = 1       # number of edges per node
p = 1
data_shots = 2  # number of measurements per problem size
batch = 3       # define how many consecutive problem sizes are evaluated before writing


# qaoa_landscape(graph, 40)  # nr_nodes, edge list, resolution

for iterator in range(1, 2):
    # c_opt(co.shgo_fun, "shgo_data.dat", range(iterator*3, (iterator+1)*3))
    # c_opt(co.nm, "nm_data.dat", range(iterator*3, (iterator+1)*3))
    # c_opt(co.bfgs, "BFGS_data.dat", range(iterator*3, (iterator+1)*3))
    # c_opt(co.rand_bfgs, "rBFGS_data.dat", range(iterator*3, (iterator+1)*3))
    # c_opt(co.direct_l, "DIRECT_L_data.dat", range(iterator*3, (iterator+1)*3))
    c_opt(co.cobyla, "COBYLA", range(iterator*batch, (iterator+1)*batch))


# TODO
#   readout success rate and runtime
#       finding maximum success rate/accuracy
#       Find runtime for set accuracy
#   scale results
#   plot?