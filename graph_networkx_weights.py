# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:38:39 2020

@author: Manos
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from functools import reduce





#create the graph
G = nx.DiGraph()
G.add_weighted_edges_from(
    [(1, 3,0.25), (2,3,0.25), (2, 5,0.75), (4, 3,1.0), (1, 8,1.0),
     (2,6,-0.25), (5, 7,-0.50), (7, 6,-1.25), (2, 7,-0.25),(5, 8,1.0),(1, 4,1.2)])

val_map = {1: 0.0,
           4: 0.6142839482638,
           5: 2.0,
           2: 1.5,
           8: 1.0}



values = [val_map.get(node, 0.25) for node in G.nodes()]

print('weight for edge 1-3')
print(G[1][3])


#number of nodes
print(G.number_of_nodes())
size=G.number_of_nodes()
print('...')



for i in range(1,8):
    print(G.degree[i])
    #print(G.out_degree[i])
    #print(G.in_degree[i])

'''
for n,nbrsdict in G.adjacency_iter():
     for nbr,eattr in nbrsdict.items():
        if 'weight' in eattr:
            (n,nbr,eattr['weight'])
G.edges(data='weight')
'''  



  

# Specify the edges we want
red_edges = [(1, 3), (2, 3),(2,5),(2,7)]
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
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
plt.show()
#exoume ka8orisei oles tis parametrous kai tupwnoume


####################################
#prints number of edges for each node
degrees = [val for (node, val) in G.degree()]
print(degrees)





#adjacency matrix
converted = nx.convert_matrix.to_numpy_matrix(G)
print(converted)


#creating matrices A+ , A- , B+ , B- , C+ , C-
#convert the matrix
Ap = np.squeeze(np.asarray(converted))
Ap[Ap < 0] = 0
print('Ap','\n',Ap)

Am = np.squeeze(np.asarray(converted))
Am[Am > 0] = 0
print('Am','\n',Am)

Aplus=np.array(Ap)
print('A+','\n',Aplus)
AplusTran=np.transpose(Aplus)
print('A+ Tranpose','\n',AplusTran)
Bplus=Aplus.dot(AplusTran)
print('B+','\n',Bplus)

Aminus=np.array(Am)
print('A-','\n',Aminus)
AminusTran=np.transpose(Aminus)
print('A- Tranpose','\n',AminusTran)
Bminus=Aminus.dot(AminusTran)
print('B-','\n',Bminus)        

Cminus=AminusTran.dot(Aminus)
print('C-','\n',Cminus)

Cplus=AplusTran.dot(Aplus)
print('C+','\n',Cplus)

