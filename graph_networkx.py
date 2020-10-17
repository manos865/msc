import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from functools import reduce

G = nx.DiGraph()
G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'E'), ('B', 'C'), ('B', 'H'),
     ('E','F'), ('B', 'G'), ('C', 'F'), ('B', 'G')])

val_map = {'A': 0.0,
           'D': 0.6142839482638,
           'H': 1.0}


values = [val_map.get(node, 0.25) for node in G.nodes()]



# Specify the edges you want here
red_edges = [('A', 'C'), ('E', 'C'),('E','F'),('C','G')]
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
#cmap=plt.get_cmap('jet')


nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
plt.show()
#exoume ka8orisei oles tis parametrous kai tupwnoume

#adjacency matrix
converted = nx.convert_matrix.to_numpy_matrix(G)
print(converted)

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


'''
for i in range(size):
    for j in range(size) :
        for k in range(size):
            Bp=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
            Bm=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
            Cp=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
            Cm=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
'''




'''
result = reduce(lambda a, x: a + x, [0]+list(range(1,3+1)))
print(result)
'''

reduce(lambda a, x: a + x, [0]+list(range(1,3+1)))



G = nx.DiGraph()
nx.add_path(G, [0, 1, 2, 3])
G.out_degree(0)  # node 0 with degree 1
list(G.out_degree([0, 1, 2]))




#from graph to matrix
A=nx.to_numpy_matrix(G)
#print(A)

G = nx.Graph([(1,1)])
A = nx.adjacency_matrix(G)

#print('Α','\n',A.todense())
A.setdiag(A.diagonal()*2) 
#print('A','\n',A.todense())

G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
G.degree[0]  # node 0 has degree 1
list(G.degree([0, 1, 2]))
#G.in_degree[0]


#G.in_degree([0,1])
#{0: 0, 1: 1}
#list(G.in_degree([0,1]).values())
