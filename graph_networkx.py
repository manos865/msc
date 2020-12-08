
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from functools import reduce



#create the graph
G = nx.DiGraph()
G.add_weighted_edges_from(
    [(1, 3,0.25), (6,1,0.75), (2,1,0.25), (3,7,0.3), (3,5,0.8), (3,2,0.25), (2,1,0.3),(7,4,0.2),
     (8,2,0.7), (2, 5,0.75), (4, 3,1.0), (1, 8,1.0), (2,8,0.3),(1,8,0.9), (1,4,0.1), (4,5,0.2),
     (2,6,-0.25), (5, 7,-0.50), (7, 6,-1.25), (2, 7,-0.25),(5, 8,1.0),(1, 4,1.2),(8,1,0.2)])

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
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
plt.show()
#exoume ka8orisei oles tis parametrous kai tupwnoume



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


print(G.out_degree[1])

'''
# create the Sums needed for the co-reference and co-citation matrices

def sigma1(i,j,k):
    s=0
    s += np.sum(Aplus[i,k]*AplusTran[k,j])
    return s

def sigma2(i,j,k):
    s=0
    s += np.sum(Aminus[i,k]*AminusTran[k,j])
    return s

def sigma3(i,j,k):
    s=0
    s += np.sum(AplusTran[k,j]*Aplus[i,k])
    return s

def sigma4(i,j,k):
    s=0
    s += np.sum(AminusTran[k,j]*Aminus[i,k])
    return s
    




#co-citation and co-reference matrices



for i in range(size):
    for j in range(size):
        for k in range(size):
            Bp=(1/(G.out_degree(i)*G.out_degree(j)))*sigma1(i, j, k)
            Bm=(1/(G.out_degree(i)*G.out_degree(j)))*sigma2(i, j, k)
            Cp=(1/(G.in_degree(i)*G.in_degree(j)))*sigma3(i, j, k)
            Cm=(1/(G.in_degree(i)*G.in_degree(j)))*sigma4(i, j, k)




#similarity and balance to be used as metrics for the affinity propagation algorithm


balance_in=[]
balance_out=[]
for i in range(size):
    for j in range(size):
        balance_in[i,j]=min(((1+Bp[i,j])/(1+Bm[i,j])),((1+Bm[i,j])/(1+Bp[i,j])))
        balance_out[i,j]=min(((1+Cp[i,j])/(1+Cm[i,j])),((1+Cm[i,j])/(1+Cp[i,j])))

Sout=[]
Sin=[]
for i in range(size):
    for j in range(size):
        Sout[i,j]=balance_out[i,j]*(Bp[i,j]+Bm[i,j])
        Sin[i,j]=balance_in[i,j]*(Cp[i,j]+Cm[i,j])

similarity=[]
for i in range(size):
    for j in range(size):
        similarity[i,j]=Sin[i,j]+Sout[i,j]



'''



'''
#example of a sigma sum
#####################################
######################################
sig=0
for i in range(size):
    for j in range(size):
        for k in range(size):
             sig += np.sum(Aplus[i,k]*AplusTran[k,j])
print(sig)

##########################################
##########################################
'''










