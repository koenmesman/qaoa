#########################################################################
# QPack                                                                 #
# Koen Mesman, TU Delft, 2021                                           #
# This file defines how classical optimizers are implemented for QAOA   #
#########################################################################
from math import pi
import matplotlib.pyplot as plt
from scipy.optimize import shgo

plt.interactive(True)
import scipy.optimize
import nlopt
import numpy as np
from qaoa_def import *


def max_cut_inv(params, graph, p):
    out = max_cut_circ(params, graph, p)
    if out == 0:
        inv = 2         # might as well be infinite or anything >1, since out >= 1
    else:
        inv = 1/out
    return inv


def max_cut_norm(params, v, e, p):
    graph = [v, e]  # graph for some reason doesn't pass normally
    out = max_cut_circ(params, graph, p)
    return out


def max_cut_alt(params, b, v, e, p):        # alternative form to use nlopt, nl_opt sends second empty param for no reason
    graph = [v, e]
    out = max_cut_circ(params, graph, p)
    return out


def tsp(params, graph, p):                  #redundant for now
    return cost_tsp(params, graph, p, 10000)


def rand_bfgs(init_param, graph, p, q_func):

    min_func = {
        'max-cut': max_cut_inv,
        'TSP' : tsp
    }.get(q_func)

    v, edge_list = graph
    best = [0]
    for i in range(2*v):
        param = np.array(np.random.uniform(low=0.0, high=2*pi, size=2))
        tmp = bfgs(param, graph, p, q_func)
        if tmp[0] > best[0]:
            best = tmp
    return best


def bobyqa(init_param, graph, p, q_func):
    v, e = graph
    min_func = {
        'max-cut': max_cut_alt,
        'TSP' : tsp
    }.get(q_func)
    opt = nlopt.opt(nlopt.LN_BOBYQA, 2)
    opt.set_max_objective(lambda a,b: min_func(a,b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-8, 1e-8]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    opt_val = opt.last_optimum_value() #returns number of expected cuts
    return [opt_val, xopt]


def direct_l(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
    opt.set_max_objective(lambda a,b: max_cut_alt(a,b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5, 1e-5]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimum_value(), xopt]


def cobyla(init_param, graph, p, q_func):

    min_func = {
        'max-cut': max_cut_alt,
        'TSP' : tsp
    }.get(q_func)

    v, e = graph
    opt = nlopt.opt(nlopt.LN_COBYLA, 2)
    opt.set_max_objective(lambda a,b: min_func(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5, 1e-5]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    opt_val = opt.last_optimum_value() #returns number of expected cuts

    return [opt_val, xopt]


def r_cobyla(init_param, graph, p):
    v, edge_list = graph
    best = 0
    for i in range(2 * v):
        param = np.array(np.random.uniform(low=0.0, high=2 * pi, size=2))
        tmp = cobyla(param, graph, p)
        if tmp[0] > best[0]:
            best = tmp
    return best



def shgo_fun(init_param, graph, p, q_func):  # simplicial homology global optimization
    bounds = [(0, 2 * pi), (0, 2 * pi)]
    res = []
    res = []
    if q_func == 'max-cut':
        min_func = max_cut_norm
        v, e = graph
        res = shgo(min_func, bounds, args=(v, e, p), options={'ftol': 1e-8}).fun  # perhaps v, e can be replaced with graph

    if q_func == 'dsp':
        min_func = dsp_cost
        v, e = graph
        res = shgo(min_func, bounds, args=(v, e, p),
                   options={'ftol': 1e-8}).fun  # perhaps v, e can be replaced with graph
    if q_func == 'tsp':
        min_func = tsp
        v, A, D = graph
        res = shgo(min_func, bounds, args=(graph, p), options={'ftol': 1e-8}).fun
    return [res.fun, res.x]


def shgo_local(init_param, graph, p, q_func):  # use this variant for a list of local optima
    bounds = [(0, 2 * pi), (0, 2 * pi)]
    res = []
    res = []
    if q_func == 'max-cut':
        min_func = max_cut_norm
        v, e = graph
        res = shgo(min_func, bounds, args=(v, e, p), options={'ftol': 1e-8}).fun  # perhaps v, e can be replaced with graph

    if q_func == 'dsp':
        min_func = dsp_cost
        v, e = graph
        res = shgo(min_func, bounds, args=(v, e, p),
                   options={'ftol': 1e-8}).fun  # perhaps v, e can be replaced with graph
    if q_func == 'tsp':
        min_func = tsp
        v, A, D = graph
        res = shgo(min_func, bounds, args=(graph, p), options={'ftol': 1e-8}).fun
    return [res.fun, res.xl]


def nm(init_param, graph, p, q_func):

    min_func = {
        'max-cut': max_cut_inv,
        'TSP' : tsp
    }.get(q_func)

    res = minimize(min_func, init_param, args=(graph, p), method='nelder-mead',
                    options={'ftol': 1e-2, 'maxfev': 400, 'disp': False})
    return [1/res.fun, res.x]

def bfgs(init_param, graph, p, q_func):
    min_func = {
        'max-cut': max_cut_inv,
        'TSP' : tsp
    }.get(q_func)
    res = minimize(min_func, init_param, args=(graph, p), method='BFGS', options={'disp': False})

    return [1/res.fun, res.x]

def g_mlsl(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GD_MLSL, 2)
    opt.set_max_objective(lambda a,b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-2, 1e-2]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimize_result(), xopt]


def g_mlsl_lds(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GD_MLSL_LDS, 2)
    opt.set_max_objective(lambda a,b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-2, 1e-2]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimize_result(), xopt]


def isres(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GN_ISRES, 2)
    opt.set_max_objective(lambda a,b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-3, 1e-3]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimize_result(), xopt]


def newuoa(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.LN_NEWUOA_BOUND, 2)
    opt.set_max_objective(lambda a, b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2*pi, 2*pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5, 1e-5]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimize_result(), xopt]


def d_annealing(init_param, graph, p):
    bounds = [(0, 2 * pi), (0, 2 * pi)]
    v, e = graph
    res = scipy.optimize.dual_annealing(max_cut_alt, bounds, args=(graph, p))
    return [res.fun, res.x]
