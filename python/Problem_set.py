#########################################################################
# Koen Mesman, TU Delft, 2021                                           #
# This file generates different problem sets for benchmarking purposes  #
#########################################################################
import time
import random
import os
import numpy as np


# returns a graph with a set edge to vertices ratio
def set_density(n, r):
    edges = []
    nr_edges = int(round(n*r))
    while nr_edges:
        tmp_edge = [random.randrange(0, n, 1), random.randrange(0, n, 1)]
        if tmp_edge[0] != tmp_edge[1]:
            tmp_edge.sort()
            if tmp_edge not in edges:
                edges.append(tmp_edge)
                nr_edges -= 1
    return edges


# return a problem graph with a set edge to vertices ratio, and ensures that all nodes are included
def include_all(n, r):
    edges = []
    nr_edges = int(round(n*r))
    for i in range(n):
        flag = True
        timeout = time.time() + 60 * 3
        retry = True
        while flag:
            while retry:
                tmp_edge = [i, random.randrange(0, n, 1)]
                if tmp_edge[0] != tmp_edge[1]:
                    tmp_edge.sort()
                    if tmp_edge not in edges:
                        edges.append(tmp_edge)
                        flag = False
                        retry = False
            if time.time() > timeout:
                print('timed out')
                break
    nr_edges -= n
    while nr_edges:
        tmp_edge = [random.randrange(0, n, 1), random.randrange(0, n, 1)]
        if tmp_edge[0] != tmp_edge[1]:
            tmp_edge.sort()
            if tmp_edge not in edges:
                edges.append(tmp_edge)
                nr_edges -= 1
    return edges


# returns the problem graph where every edge has a set probability
def set_probability(n, p):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                edges.append([i, j])
    return edges


# return a regular graph
def regular_graph(n):
    edges = []
    for i in range(n-1):
        edges.append([i, i+1])
    edges.append([0, n-1])
    for i in range(n-2):
        edges.append([i, i+2])
    edges.append([0, n-2])
    edges.append([1, n-1])
    return edges


# return a fully connected graph (set_probability p=1) if weighted == True, give each edge a weight [1, 10]
def fully_connected(n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j])
    return edges


def create_dir(new_dir):
    path = "problem_sets"
    try:
        if not os.path.exists(path):
            os.makedirs(path, 0o700)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    path = "problem_sets" + str(new_dir)
    try:
        if not os.path.exists(path):
            os.makedirs(path, 0o700)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


# transforms the graph to a weighted graph with edge value inf for disconnected vertices
def to_tsp(edge_list):
    n = max(edge_list, key=lambda x: x[1])[1]+1
    full_list = fully_connected(n)
    for i in range(len(full_list)):
        if full_list[i] not in edge_list:
            full_list[i].append(np.inf)
        else:
            full_list[i].append(random.randrange(1, 10, 1))
    return full_list


# generate a file with size 5 to n vertices with selected method
def generate_problem(method, n, *arg):

    create_dir("/"+method.__name__)
    edges = []

    if len(arg) == 1:
        for i in range(5, n + 1):
            p = arg[0]
            edges = method(n, p)
            path = "problem_sets/"+method.__name__+"/test_" + str(i) + ".dat"
            np.savetxt(path, edges)

    else:
        for i in range(5, n+1):

            edges = method(n)
            path = "problem_sets/"+method.__name__+"/test_" + str(i) + ".dat"
            np.savetxt(path, edges)

    return edges


# examples:
graph = generate_problem(regular_graph, 6)
tsp_graph = to_tsp(graph)
