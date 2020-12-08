# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:38:39 2020

"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from functools import reduce
import copy


#create the graph
G = nx.DiGraph()
G.add_weighted_edges_from(
    [(1, 3,0.25), (2,1,0.75), (3,2,0.3), (2,1,-0.75), (3,2,-0.3)
    (1,2,0.75), (2,3,0.3) , (3,1,0.75), (3,2,0.3)])

val_map = {1: 0.0,
           4: 0.6142839482638,
           5: 2.0,
           2: 1.5,
           8: 1.0}



values = [val_map.get(node, 0.25) for node in G.nodes()]

#print('weight for edge 1-3')
#print(G[1][3])


#number of nodes
#print(G.number_of_nodes())
size=G.number_of_nodes()
#print('...')


#prints number of edges (in and out) for each node
#for i in range(1,8):
    #print(G.degree[i])
    #print(G.out_degree[i])
    #print(G.in_degree[i])
  

# Specify the edges we want
red_edges = [(1, 3), (3, 2)]
edge_colours = ['black' if not edge in red_edges else 'red'
                for edge in G.edges()]
black_edges = [edge for edge in G.edges() if edge not in red_edges]


# Need to create a layout when doing
# separate calls to draw nodes and edges
pos = nx.spring_layout(G)
#orizei to layout gia thn optikopoihsh tou G grafou pou orisame parapanw
#pos:Initial positions for nodes as a dictionary with node as keys and values as a list or tuple.
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 500)
cmap=plt.get_cmap('jet')

nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
plt.show()
#exoume ka8orisei oles tis parametrous kai tupwnoume


####################################
#prints number of edges for each node
degrees = [val for (node, val) in G.degree()]
#print(degrees)


#adjacency matrix
converted = nx.convert_matrix.to_numpy_matrix(G)
print(converted)

#creating matrices A+ , A- , B+ , B- , C+ , C-
#convert the matrix
c = copy.deepcopy(converted)
Ap = np.squeeze(np.asarray(c))
Ap[Ap < 0] = 0
print('Ap','\n',Ap)


c = copy.deepcopy(converted)
Am = np.squeeze(np.asarray(c))
Am[Am > 0] = 0
print('Am','\n',Am)


Aplus = np.array(Ap)
print('A+','\n',Aplus)
AplusTran = np.transpose(Aplus)
print('A+ Tranpose','\n',AplusTran)
Bplus = Aplus.dot(AplusTran)
print('B+','\n',Bplus)

Aminus = np.array(Am)
print('A-','\n',Aminus)
AminusTran = np.transpose(Aminus)
print('A- Tranpose','\n',AminusTran)
Bminus = Aminus.dot(AminusTran)
print('B-','\n',Bminus)        

Cminus = AminusTran.dot(Aminus)
print('C-','\n',Cminus)

Cplus = AplusTran.dot(Aplus)
print('C+','\n',Cplus)


node_degrees = []

for node in G.nodes():
        
    degree_map = dict()
    degree_map['out'] = dict()
    degree_map['in'] = dict()
    
    degree_map['out']['positive'] = 0
    degree_map['out']['negative'] = 0
    degree_map['in']['positive'] = 0
    degree_map['in']['negative'] = 0

    for (edge_src,edge_dst,d) in G.edges(data=True):
        #print('edge source :', edge_src)
        #print('edge destination :', edge_dst)
        edge_weight = d['weight']
        #print('weight:',edge_weight)
        if edge_src == node:
            if edge_weight > 0.5:
                degree_map['out']['positive'] += 1
            else:
                degree_map['out']['negative'] += 1
        elif edge_dst == node:
            if edge_weight > 0.5:
                degree_map['in']['positive'] += 1
            else:
                degree_map['in']['negative'] += 1                
    
    print('node:',node)
    print('out positive',degree_map['out']['positive'])
    print('in positive',degree_map['in']['positive'])
    print('out negative',degree_map['out']['negative'])
    print('in negative',degree_map['in']['negative'])        
    node_degrees.append(degree_map)

print(node_degrees)
print(node_degrees[2]['in']['negative'])





# create the Sums needed for the co-reference and co-citation matrices

def calculate_sigma1(i,j,k):
    s=0
    for c in range(k):
        s += np.sum(Aplus[i,k]*AplusTran[k,j])
    return s
    


#co-citation and co-reference matrices

Bp = [[0 for i in range(size)] for j in range(size)]
print(Bp)

for i in range(size):
    for j in range(size):
        denominator = (node_degrees[i]['out']['positive'])*(node_degrees[j]['out']['positive'])
        if denominator == 0:
            Bp[i][j] = 0
        else :
            sigma = calculate_sigma1(i,j,2)
            Bp[i][j] = (1/(denominator))*sigma

print('Bp:',Bp)




        
        
        
        