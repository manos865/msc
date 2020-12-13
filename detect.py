"""Community detection in signed directed graphs."""

import copy
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation


def generate_directed_signed_graph(n: int, p: float = 0.015,
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


def get_adjacency_matrix(g):
    """Return the adjacency matrix of the specified graph.

    Args:
        g: The graph for which to calculate the adjacency matrix

    Returns:
        The adjacency matrix of g
    """
    return nx.to_numpy_matrix(g)


def calculate_node_degrees(g):
    """Calculate all node degrees for the specified graph.

    Return a list that contains the following degrees for a each node:

    1. out-positive degree: number of outgoing edges, with positive sign
    2. out-negative degree: number of outgoing edges, with negative sign
    3. in-positive degree: number of incoming edges, with positive sign
    4. in-negative degree: number of incoming edges, with negative sign

    The length of the returned list equals the size of the graph.
    """
    node_degrees = []

    for node in g.nodes():
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
    g = None
    size = 3000
    accuracy = 3

    degrees = [0]
    while 0 in degrees:
        g = generate_directed_signed_graph(n=size)
        degrees = [val for (node, val) in g.degree()]

    print("Number of nodes %s" % g.number_of_nodes())
    print("Number of edges: %s" % g.number_of_edges())
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

    # Transpose of positive adjacency matrix A+
    adj_mat_pos_trans = np.transpose(adj_mat_pos)

    # Transpose of negative adjacency matrix A-
    adj_mat_neg_trans = np.transpose(adj_mat_neg)


    print("Adjacency Matrix (A)\n", adj_mat, "\n")
    print("Positive Adjacency Matrix (A+)\n", adj_mat_pos, "\n")
    print("Negative Adjacency Matrix (A-)\n", adj_mat_neg, "\n")
    
    print("Transpose of Positive Adjacency Matrix (A+)\n", adj_mat_pos_trans)
    print("\n")
    print("Transpose of Negative Adjacency Matrix (A-)\n", adj_mat_neg_trans)
    print("\n")






    #calculate node degrees
    node_degrees = calculate_node_degrees(g)







    # Co-reference matrix B+
    # Contains the number of nodes that are commonly cited with positive sign
    # by two nodes
    b_pos = np.matmul(adj_mat_pos, adj_mat_pos_trans)
    for i in range(size):
        for j in range(size):
            norm_factor = (node_degrees[i]["out"]["positive"] *
                           node_degrees[j]["out"]["positive"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of positive out-degrees is 0, setting positive"
                      " co-reference of (%d,%d) to 0" % (i, j))
            b_pos[i, j] = norm_factor * b_pos[i, j]
    b_pos = np.round(b_pos, decimals=accuracy)

    # Co-reference matrix B-
    # Contains the number of nodes that are commonly cited with negative sign
    # by two nodes
    b_neg = np.matmul(adj_mat_neg, adj_mat_neg_trans)
    for i in range(size):
        for j in range(size):
            norm_factor = (node_degrees[i]["out"]["negative"] *
                           node_degrees[j]["out"]["negative"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of negative out-degrees is 0, setting negative"
                      " co-reference of (%d,%d) to 0" % (i, j))
            b_neg[i, j] = norm_factor * b_neg[i, j]
    b_neg = np.round(b_neg, decimals=accuracy)

    # Positive Co-Citation matrix C+
    # Contains the number of nodes that commonly point to both i and j with
    # positive sign
    c_pos = np.matmul(adj_mat_pos_trans, adj_mat_pos)
    for i in range(size):
        for j in range(size):
            norm_factor = (node_degrees[i]["in"]["positive"] *
                           node_degrees[j]["in"]["positive"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of positive in-degrees is 0, setting positive"
                      " co-citation of (%d,%d) to 0" % (i, j))
            c_pos[i, j] = norm_factor * c_pos[i, j]
    c_pos = np.round(c_pos, decimals=accuracy)

    # Negative Co-Citation matrix C-
    # Contains the number of nodes that commonly point to both i and j with
    # negative sign
    c_neg = np.matmul(adj_mat_neg_trans, adj_mat_neg)
    for i in range(size):
        for j in range(size):
            norm_factor = (node_degrees[i]["in"]["negative"] *
                           node_degrees[j]["in"]["negative"])
            if norm_factor != 0:
                norm_factor = 1 / norm_factor
            else:
                print("Product of negative in-degrees is 0 - setting negative"
                      " co-citation of (%d,%d) to 0" % (i, j))
            c_neg[i, j] = norm_factor * c_neg[i, j]
    c_neg = np.round(c_neg, decimals=accuracy)


    print("Positive Co-Reference Matrix (B+)\n", b_pos, "\n")
    print("Negative Co-Reference Matrix (B-)\n", b_neg, "\n")
    print("Positive Co-Citation Matrix (C+)\n", c_pos, "\n")
    print("Negative Co-Citation Matrix (C-)\n", c_neg, "\n")








    # Balance of incoming and outgoing links
    balance_in = np.zeros([size, size])
    balance_out = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            balance_in[i, j] = min((1 + b_pos[i, j]) / (1 + b_neg[i, j]),
                                   (1 + b_neg[i, j]) / (1 + b_pos[i, j]))
            balance_out[i, j] = min((1 + c_pos[i, j]) / (1 + c_neg[i, j]),
                                    (1 + c_neg[i, j]) / (1 + c_pos[i, j]))

    # Similarity based on incoming and outgoing links
    sim_in = np.add(b_pos, b_neg)
    sim_out = np.add(c_pos, c_neg)
    for i in range(size):
        for j in range(size):
            sim_in[i, j] *= balance_in[i, j]
            sim_out[i, j] *= balance_out[i, j]

    similarity = np.add(sim_in, sim_out)


    print("In-link Balance Matrix\n", balance_in, "\n")
    print("Out-link Balance Matrix\n", balance_out, "\n")
    print("Incoming Link Similarity Matrix\n", sim_in, "\n")
    print("Outgoing Link Similarity Matrix\n", sim_out, "\n")
    print("Total Similarity Matrix\n", similarity, "\n")








    # Run the Affinity Propagation algorithm
    af = AffinityPropagation(affinity="precomputed", verbose=True,
                             random_state=0)
    af.fit(similarity)

    clusters = []
    number_of_clusters = len(af.cluster_centers_indices_)
    for i in range(number_of_clusters):
        clusters.append([])

    for i in range(size):
        for j in range(number_of_clusters):
            if af.labels_[i] == j:
                clusters[j].append(i + 1)

    print("\nNumber of clusters: %d\n" % number_of_clusters)
    print("Cluster centers: %s\n" % af.cluster_centers_indices_)
    print("Clusters: %s\n" % clusters)




    # Visualize the original graph
    # visualize_graph(g)




if __name__ == "__main__":
    main()
