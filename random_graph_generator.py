# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:18:00 2020

@author: Manos
"""

"""Community detection in signed directed graphs."""

import copy
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_directed_signed_graph(n: int, p: float = 0.25,
                                   weights: list = None) -> nx.DiGraph:
    """Generate a random directed signed graph.

    Each node can have the following edges:

    1) positive outgoing/referring (w=1)
    2) negative outgoing/referring (w=-1)
    3) positive incoming/citing (w=1)
    4) negative incoming/citing (w=-1)

    Args:
        n: Number of nodes
        p: Probability of two nodes being connected (between 0 and 1)

    Returns:
        g: A random directed signed graph
    """
    print("Generating random signed directed graph with %d nodes" % n)
    g = nx.gnp_random_graph(n, p, directed=True)
    edges = g.edges(data=True)

    # Sign edges: 1 = postive, -1 = negative
    if weights is None:
        weights = [-1, 1]

    for i, (u, v, d) in enumerate(edges):
        d["weight"] = random.choice(weights)
        d["color"] = "green" if d["weight"] > 0 else "red"

    return g


def visualize_graph(g):
    """Visualize the specified graph."""
    pos = nx.spring_layout(g)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_nodes(g, pos, node_size=150)
    colors = [g[u][v]["color"] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, arrows=True, edge_color=colors)
    plt.show()


def get_adjacency_matrix(g):
    """Return the adjacency matrix of the specified graph.

    Args:
        g: The graph for which to calculate the adjacency matrix

    Returns:
        The adjacency matrix of g
    """
    return nx.to_numpy_matrix(g)


def main():
    """Run algorithm on random graph."""
    g = None
    degrees = [0]
    while 0 in degrees:
        g = generate_directed_signed_graph(n=50)
        degrees = [val for (node, val) in g.degree()]

    print("Number of nodes %s\n" % g.number_of_nodes())
    print("Number of edges: %s\n" % g.number_of_edges())
    print("Degrees: %s\n" % degrees)

    # Adjacency matrix A
    adj_mat = get_adjacency_matrix(g)

    # Positive adjacency matrix A+
    adj_mat_pos = copy.deepcopy(adj_mat)
    adj_mat_pos = np.asarray(adj_mat_pos)
    adj_mat_pos[adj_mat_pos <= 0] = 0

    # Negative adjacency matrix A-
    adj_mat_neg = copy.deepcopy(adj_mat)
    adj_mat_neg = np.asarray(adj_mat_neg)
    adj_mat_neg[adj_mat_neg >= 0] = 0

    print("Adjacency Matrix A\n", adj_mat, "\n")
    print("Positive Adjacency Matrix A+\n", adj_mat_pos, "\n")
    print("Negative Adjacency Matrix A-\n", adj_mat_neg, "\n")

    # Visualize the graph
    visualize_graph(g)


if __name__ == "__main__":
    main()