from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import shgo

plt.interactive(True)
import scipy.optimize
import nlopt
import numpy as np
from qaoa_def import *


def qaoa_inv(params, graph, p):
    out = qaoa_circ(params, graph, p)
    if out == 0:
        inv = 2         # might as well be infinite or anything >1, since out >= 1
    else:
        inv = 1/out
    return inv


def qaoa_norm(params, v, e, p):
    graph = [v, e]  # graph for some reason doesn't pass normally
    out = qaoa_circ(params, graph, p)
    return out


def qaoa_alt(params, b, v, e, p):        # alternative form to use nlopt, nl_opt sends second empty param for no reason
    graph = [v, e]
    out = qaoa_circ(params, graph, p)
    return out


def rand_bfgs(init_param, graph, p):
    v, edge_list = graph
    best = 2
    for i in range(2*v):
        param = np.array(np.random.uniform(low=0.0, high=2*pi, size=2))
        tmp = minimize(qaoa_inv, param, args=(graph, p), method='BFGS',
                 options={'disp': False})
        if tmp < best:
            best = tmp
    return best


def bobyqa(init_param, graph, p):
    v, e = graph
    global g_graph
    g_graph = graph
    opt = nlopt.opt(nlopt.LN_BOBYQA, 2)
    opt.set_max_objective(lambda a,b: qaoa_alt(a,b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-8, 1e-8]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return xopt


def direct_l(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
    opt.set_max_objective(lambda a,b: qaoa_alt(a,b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5, 1e-5]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return xopt


def cobyla(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.LN_COBYLA, 2)
    opt.set_max_objective(lambda a,b: qaoa_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5, 1e-5]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return xopt


def shgo_fun(init_param, graph, p):
    bounds = [(0, 2 * pi), (0, 2 * pi)]
    v, e = graph
    shgo(qaoa_norm, bounds, args=(v, e, p))


def nm(init_param, graph, p):
    return minimize(qaoa_inv, init_param, args=(graph, p), method='nelder-mead',
                    options={'ftol': 1e-2, 'maxfev': 400, 'disp': False})


def bfgs(init_param, graph, p):
    return minimize(qaoa_inv, init_param, args=(graph, p), method='BFGS', options={'disp': False})