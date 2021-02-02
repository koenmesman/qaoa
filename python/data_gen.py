from math import pi
import time
import random
from matplotlib import cm
import matplotlib.pyplot as plt
plt.interactive(True)
#backend = Aer.get_backend('qasm_simulator')
from scipy.optimize import minimize
import nlopt
import numpy as np
import Classic_opt as co
from qaoa_def import qaoa_landscape

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


def grid_search(graph, n):
    min_val = []
    X, Y, val = qaoa_landscape(graph, n, p, False)
    for i in X:
        for j in Y:
            if val[i][j] < min:
                min_val = val[i][j]
    best = 1 / min_val
    return best


def test_graph():
    v = 3
    e = [[0, 1, 5], [1, 2, 8], [0, 2, 7]]
    A = [[0 for x in range(v)] for x in range(v)]
    D = [[0 for x in range(v)] for x in range(v)]

    w = []
    for i in range(len(e)):
        w.append(e[i][2])
    for n in range(v):
        D[n][n] = 2*max([item[2] for item in e] )
    for t in e:
        A[t[0]][t[1]] = 1
        A[t[1]][t[0]] = 1
        D[t[0]][t[1]] = t[2]
        D[t[1]][t[0]] = t[2]
    A = [item for sublist in A for item in sublist]
    D = [item for sublist in D for item in sublist]
    return v, A, D


def tsp_graph(v, r):
    A = [[0 for x in range(v)] for x in range(v)]
    D = [[0 for x in range(v)] for x in range(v)]

    nodes = list(range(0, v))
    e = int(round(v*r))
    edges = [[0]*3 for _ in range(e)]
    super_flag = 1
    timeout = time.time() + 60 * 5
    while super_flag:
        for x in range(e):
            flag = 1
            while flag:
                node_a = nodes
                edges[x][0] = random.choice(node_a[:(v-1)])
                node_b = nodes
                edges[x][1] = random.choice(node_b[edges[x][0] + 1:])
                edges[x][2] = random.randint(1, 10)
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

    e = edges
    w = []
    for i in range(len(e)):
        w.append(e[i][2])
    for n in range(v):
        D[n][n] = 2*max([item[2] for item in e] )
    for t in e:
        A[t[0]][t[1]] = 1
        A[t[1]][t[0]] = 1
        D[t[0]][t[1]] = t[2]
        D[t[1]][t[0]] = t[2]
    A = [item for sublist in A for item in sublist]
    D = [item for sublist in D for item in sublist]
    return v, A, D


def acc_test(method_tuple):
    for problem_size in range(3, lim_range+1):
        # data_x = [problem_size]*3

        for d in range(data_shots):  # loop over data points per problem size, time optimizer
            print("initiated problem size ", problem_size)
            graph = generate_graph(problem_size, ratio)
            best = co.d_annealing([pi, pi], graph, p)

            for ituple in method_tuple:
                # d_acc = []
                opt_func, data_name = ituple
                print("begin method ", data_name)

                init_param = [pi, pi]
                d_acc = []
                data_x = []
                for k in range(2):
                    result = opt_func(init_param, graph, p)
                    print("now finished iteration", d, " for method ", data_name)
                    #d_acc.append(result)
                    d_acc.append((best - result)/result)
                    data_x.append(problem_size)
                data_path = 'Test_data/Accuracy/' + data_name + '_rel_acc_data.dat'
                try:
                    load_data = np.loadtxt(data_path)
                    save_data = np.array([data_x, d_acc])
                    print(save_data)
                    data = np.hstack((load_data, save_data))
                except IOError:
                    data = np.array([data_x, d_acc])
                finally:
                    np.savetxt(data_path, data)
                    print("data saved for method " + data_name + " for problem size ", problem_size)
    print('Done')



######################################################################################
ratio = 1       # number of edges per node
p = 1
data_shots = 2  # number of measurements per problem size
batch = 3       # define how many consecutive problem sizes are evaluated before writing
lim_range = 20

# qaoa_landscape(graph, 40)  # nr_nodes, edge list, resolution

#for iterator in range(1, 2):
    # c_opt(co.shgo_fun, "shgo_e-8", range(iterator*3, (iterator+1)*3))
    #c_opt(co.nm, "nm", range(iterator*3, (iterator+1)*3))
    # c_opt(co.bfgs, "BFGS", range(iterator*3, (iterator+1)*3))
    # c_opt(co.rand_bfgs, "rBFGS", range(iterator*3, (iterator+1)*3))
    # c_opt(co.cobyla, "COBYLA", range(iterator*batch, (iterator+1)*batch))
    #c_opt(co.isres, "ISRES", range(iterator*batch, (iterator+1)*batch))
    #c_opt(co.newuoa, "NEWUOA", range(iterator*batch, (iterator+1)*batch))
#    c_opt(co.g_mlsl, "G_MLSL", range(iterator*batch, (iterator+1)*batch))
    #c_opt(co.g_mlsl_lds, "G_MLSL_LDS", range(iterator*batch, (iterator+1)*batch))
    # c_opt(co.direct_l, "DIRECT_L", range(iterator*3, (iterator+1)*3))
    #c_opt(co.bobyqa, "BOBYQA_e8", range(iterator*batch, (iterator+1)*batch))


#method_arr = [co.shgo_fun, co.nm, co.bfgs, co.cobyla, co.bobyqa, co.r_cobyla]
#data_arr = ['shgo', 'nm', 'bfgs', 'cobyla', 'bobyqa', 'r_cobyla']

#method_arr = [co.shgo_fun]
#data_arr = ['shgo']

#method_tuple = [0]*len(data_arr)
#for i in range(len(data_arr)):
#    method_tuple[i] = [method_arr[i], data_arr[i]]

#start = time.time()
#acc_test(method_tuple)
#print('This run took ', (time.time() - start), 'seconds')

# TODO
#   readout success rate and runtime
#       finding maximum success rate/accuracy
#       Find runtime for set accuracy
#   scale results
#   plot?