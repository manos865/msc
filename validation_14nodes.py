# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 23:07:04 2021

@author: Manos
"""

"""Community detection in signed directed graphs."""

import copy
import random
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation


SIZE = None
CONNECTIVITY = None
ACCURACY = 3



def generate_directed_signed_graph_g1():
    """Generate hardcoded graph 1."""
    g = nx.DiGraph()

    g.add_edge(5, 0, weight=1)
    g.add_edge(5, 1, weight=1)
    g.add_edge(5, 2, weight=1)
    g.add_edge(5, 3, weight=1)
    g.add_edge(5, 4, weight=1)

    g.add_edge(6, 0, weight=1)
    g.add_edge(6, 1, weight=1)
    g.add_edge(6, 2, weight=1)
    g.add_edge(6, 3, weight=1)
    g.add_edge(6, 4, weight=1)

    g.add_edge(5, 6, weight=1)
    g.add_edge(7, 8, weight=1)
    
    g.add_edge(7, 9, weight=-1)
    g.add_edge(7, 10, weight=-1)
    g.add_edge(7, 11, weight=-1)
    g.add_edge(7,12,weight=-1)
    g.add_edge(7,13,weight=-1)
    
    g.add_edge(8, 9, weight=-1)
    g.add_edge(8, 10, weight=-1)
    g.add_edge(8, 11, weight=-1)
    g.add_edge(8,12,weight=-1)
    g.add_edge(8,13,weight=-1)
    
    for i, (u, v, d) in enumerate(g.edges(data=True)):
        d["color"] = "green" if d["weight"] > 0 else "red"

    return g


def generate_directed_signed_graph(n: int, p: float,
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
    print("Generating random signed directed graph with %d nodes and %.2f%%"
          " connectivity" % (n, 100*p))
    g = nx.gnp_random_graph(n, p, directed=True)
    edges = g.edges(data=True)

    # Sign edges: 1 = positive, -1 = negative
    if weights is None:
        weights = [-1, 1]

    for i, (u, v, d) in enumerate(edges):
        d["weight"] = random.choice(weights)

    g = nx.DiGraph()

    g.add_edge(0, 5, weight=1)
    g.add_edge(1, 5, weight=1)
    g.add_edge(2, 5, weight=1)
    g.add_edge(3, 5, weight=1)
    g.add_edge(4, 5, weight=1)

    g.add_edge(0, 6, weight=-1)
    g.add_edge(1, 6, weight=-1)
    g.add_edge(2, 6, weight=-1)
    g.add_edge(3, 6, weight=-1)
    g.add_edge(4, 6, weight=-1)

    for i, (u, v, d) in enumerate(g.edges(data=True)):
        d["color"] = "green" if d["weight"] > 0 else "red"
        # print(u, v, d)

    return g


def visualize_graph(g):
    """Visualize the specified graph."""
    pos = nx.spring_layout(g)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_nodes(g, pos, node_size=250)
    colors = [g[u][v]["color"] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, arrows=True, edge_color=colors)
    plt.show()


def get_adjacency_matrix(g,SIZE):
    """Return the adjacency matrix of the specified graph.

    Args:
        g: The graph for which to calculate the adjacency matrix

    Returns:
        The adjacency matrix of g
    """
    return nx.to_numpy_matrix(g,nodelist=range(SIZE))


def calculate_node_degrees(g):
    """Calculate all node degrees for the specified graph.

    Return a list that contains the following degrees for a each node:

    1. out-positive degree: number of outgoing edges, with positive sign
    2. out-negative degree: number of outgoing edges, with negative sign
    3. in-positive degree: number of incoming edges, with positive sign
    4. in-negative degree: number of incoming edges, with negative sign

    The length of the returned list equals the SIZE of the graph.
    """
    node_degrees = []

    for node in sorted(g.nodes()):
        degree_map = dict()
        degree_map["out"] = dict()
        degree_map["in"] = dict()

        degree_map["out"]["positive"] = 0
        degree_map["out"]["negative"] = 0
        degree_map["in"]["positive"] = 0
        degree_map["in"]["negative"] = 0

        for (src, dst, d) in g.edges(data=True):
            edge_weight = d["weight"]
            if src == node:
                if int(edge_weight) > 0:
                    degree_map["out"]["positive"] += 1
                else:
                    degree_map["out"]["negative"] += 1
            elif dst == node:
                if int(edge_weight) > 0:
                    degree_map["in"]["positive"] += 1
                else:
                    degree_map["in"]["negative"] += 1

        node_degrees.append(degree_map)

        # print("node:", node)
        # print("out positive:", degree_map["out"]["positive"])
        # print("in positive:", degree_map["in"]["positive"])
        # print("out negative:", degree_map["out"]["negative"])
        # print("in negative:", degree_map["in"]["negative"])
        # print("\n")

    return node_degrees


def main():
    """Run algorithm on random graph."""
    parser = argparse.ArgumentParser(description="input for connectivity and"
                                                 " size arguments.")
    parser.add_argument("size", type=int)
    parser.add_argument("connectivity", type=float)
    args = parser.parse_args()

    global CONNECTIVITY, SIZE
    CONNECTIVITY = args.connectivity
    SIZE = args.size

    g = None

    # degrees = [0]
    # while 0 in degrees:
    #     g = generate_directed_signed_graph(n=SIZE, p=CONNECTIVITY)
    #     degrees = [val for (node, val) in g.degree()]

    g = generate_directed_signed_graph_g1()
    print("Number of nodes %s" % g.number_of_nodes())
    print("Number of edges: %s" % g.number_of_edges())
    # print("Degrees: %s\n" % degrees)

    # Adjacency matrix A

    #A = nx.adjacency_matrix(g,nodelist=range(SIZE))
    #print(A.todense())
    adj_mat = get_adjacency_matrix(g,SIZE)
    print(adj_mat)
    # Positive adjacency matrix A+
    adj_mat_pos = copy.deepcopy(adj_mat)
    adj_mat_pos = np.asarray(adj_mat_pos)
    adj_mat_pos[adj_mat_pos <= 0] = 0

    # Negative adjacency matrix A-
    adj_mat_neg = copy.deepcopy(adj_mat)
    adj_mat_neg = np.asarray(adj_mat_neg)
    adj_mat_neg[adj_mat_neg >= 0] = 0

    print("Adjacency Matrix (A)\n", adj_mat, "\n")
    print("Positive Adjacency Matrix (A+)\n", adj_mat_pos, "\n")
    print("Negative Adjacency Matrix (A-)\n", adj_mat_neg, "\n")

    # Transpose of positive adjacency matrix A+
    adj_mat_pos_trans = np.transpose(adj_mat_pos)

    # Transpose of negative adjacency matrix A-
    adj_mat_neg_trans = np.transpose(adj_mat_neg)

    print("Transpose of Positive Adjacency Matrix (A+)\n", adj_mat_pos_trans,
          "\n")
    print("Transpose of Negative Adjacency Matrix (A-)\n", adj_mat_neg_trans,
          "\n")

    node_degrees = calculate_node_degrees(g)

    # Co-reference matrix B+
    # Contains the number of nodes that are commonly cited with positive sign
    # by two nodes
    b_pos = np.matmul(adj_mat_pos, adj_mat_pos_trans)
    '''
    for i in range(SIZE):
        for j in range(SIZE):
            norm_factor = (node_degrees[i]["out"]["positive"] *
                           node_degrees[j]["out"]["positive"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of positive out-degrees is 0, setting positive"
                      " co-reference of (%d,%d) to 0" % (i, j))
            b_pos[i, j] = norm_factor * b_pos[i, j]     
    '''
    b_pos = np.round(b_pos, decimals=ACCURACY)

    # Co-reference matrix B-
    # Contains the number of nodes that are commonly cited with negative sign
    # by two nodes
    b_neg = np.matmul(adj_mat_neg, adj_mat_neg_trans)
    '''
    for i in range(SIZE):
        for j in range(SIZE):
            norm_factor = (node_degrees[i]["out"]["negative"] *
                           node_degrees[j]["out"]["negative"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of negative out-degrees is 0, setting negative"
                      " co-reference of (%d,%d) to 0" % (i, j))
            b_neg[i, j] = norm_factor * b_neg[i, j]
    '''
    b_neg = np.round(b_neg, decimals=ACCURACY)

    # Positive Co-Citation matrix C+
    # Contains the number of nodes that commonly point to both i and j with
    # positive sign
    c_pos = np.matmul(adj_mat_pos_trans, adj_mat_pos)
    '''
    for i in range(SIZE):
        for j in range(SIZE):
            norm_factor = (node_degrees[i]["in"]["positive"] *
                           node_degrees[j]["in"]["positive"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of positive in-degrees is 0, setting positive"
                      " co-citation of (%d,%d) to 0" % (i, j))
            c_pos[i, j] = norm_factor * c_pos[i, j]
    '''
    c_pos = np.round(c_pos, decimals=ACCURACY)

    # Negative Co-Citation matrix C-
    # Contains the number of nodes that commonly point to both i and j with
    # negative sign
    c_neg = np.matmul(adj_mat_neg_trans, adj_mat_neg)
    '''
    for i in range(SIZE):
        for j in range(SIZE):
            norm_factor = (node_degrees[i]["in"]["negative"] *
                           node_degrees[j]["in"]["negative"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of negative in-degrees is 0 - setting negative"
                      " co-citation of (%d,%d) to 0" % (i, j))
            c_neg[i, j] = norm_factor * c_neg[i, j]
    '''
    c_neg = np.round(c_neg, decimals=ACCURACY)

    print("Positive Co-Reference Matrix (B+)\n", b_pos, "\n")
    print("Negative Co-Reference Matrix (B-)\n", b_neg, "\n")
    print("Positive Co-Citation Matrix (C+)\n", c_pos, "\n")
    print("Negative Co-Citation Matrix (C-)\n", c_neg, "\n")

    # Balance of incoming and outgoing links
    balance_in = np.zeros([SIZE, SIZE])
    balance_out = np.zeros([SIZE, SIZE])
    for i in range(SIZE):
        for j in range(SIZE):
            balance_out[i, j] = min((1 + b_pos[i, j]) / (1 + b_neg[i, j]),
                                   (1 + b_neg[i, j]) / (1 + b_pos[i, j]))
            balance_in[i, j] = min((1 + c_pos[i, j]) / (1 + c_neg[i, j]),
                                    (1 + c_neg[i, j]) / (1 + c_pos[i, j]))
    
    balance_in = np.round(balance_in, decimals=ACCURACY)
    balance_out = np.round(balance_out, decimals=ACCURACY)

    print("Incoming Link Balance Matrix\n", balance_in, "\n")
    print("Outgoing Link Balance Matrix\n", balance_out, "\n")

    # Similarity based on incoming and outgoing links
    sim_out = np.add(b_pos, b_neg)
    sim_in = np.add(c_pos, c_neg)

    #sim_out = np.matmul(balance_in, sim_in)
    #sim_in = np.matmul(balance_in, sim_out)

    sim_out = np.round(sim_out, decimals=ACCURACY)
    sim_in = np.round(sim_in, decimals=ACCURACY)

    print("Incoming Link Similarity Matrix\n", sim_in, "\n")
    print("Outgoing Link Similarity Matrix\n", sim_out, "\n")

    similarity = np.add(sim_in, sim_out)
    print("Total Similarity Matrix\n", similarity, "\n")


    #print diagonal values
    diagonal = []
    for i in range(SIZE):
        for j in range(SIZE):
            if i == j:
                diagonal.append(similarity[i, j])
    print("Diagonal values\n", diagonal, "\n")



    

    #check node degrees to be sorted
    #############################################
    #############################################
    #############################################
    #Number of Total Edges per node
    nte = []
    for i in range(SIZE):
        nte.append(node_degrees[i]["out"]["positive"] + node_degrees[i]["out"]["negative"] + node_degrees[i]["in"]["positive"] + node_degrees[i]["in"]["negative"])
    #for i in range(SIZE):
        #print('node:',i,' has ',nte[i],' edges.')
        
    #number of maximum edges per pair of nodes matrix
    M=np.zeros((SIZE,SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            M[i,j]=max(nte[i],nte[j])
    print('\n Matrix of maximum number of edges per pair of nodes \n')
    print(M)

    ################################
    #similarity as percentage matrix
    SP = np.zeros((SIZE,SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            SP[i,j]=similarity[i,j]/M[i,j]
    print('\n Final Similarity matrix \n')
    np.set_printoptions(linewidth=np.inf)
    print(SP)
    #print diagonal values of total similarity matrix
    diagonal = []
    for i in range(SIZE):
        for j in range(SIZE):
            if i == j:
                diagonal.append(SP[i, j])
    print("Diagonal values\n", diagonal, "\n")
    ##############################################
    ##############################################
    ##############################################




    


    # Run the Affinity Propagation algorithm
    af = AffinityPropagation(affinity="precomputed", verbose=True,
                             random_state=0)
    af.fit(SP)
    #af.fit(sim_in)
    #af.fit(sim_out)

    clusters = []
    number_of_clusters = len(af.cluster_centers_indices_)
    for i in range(number_of_clusters):
        clusters.append([])

    for i in range(SIZE):
        for j in range(number_of_clusters):
            if af.labels_[i] == j:
                clusters[j].append(i)

    print("\nNumber of clusters: %d\n" % number_of_clusters)
    print("Cluster centers: %s\n" % af.cluster_centers_indices_)
    print("Clusters: %s\n" % clusters)

    # Visualize the original graph
    visualize_graph(g)

    



if __name__ == "__main__":
    main()