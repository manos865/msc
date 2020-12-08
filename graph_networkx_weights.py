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
#    [(1,3,1),(1,2,1),(3,1,-1),(2,1,-1),(2,3,1),(3,2,1)]
     [(1, 3,1), (6,1,1), (2,1,-1), (3,7,1), (3,5,1), (3,2,-1), (2,1,1),(7,4,-1),
     (8,2,1), (2, 5,-1), (4, 3,-1), (1, 8,-1), (2,8,1),(1,8,-1), (1,4,1), (4,5,-1),
     (2,6,1), (5, 7,-1), (7, 6,-1), (2, 7,-1),(5, 8,1),(1, 4,1),(8,1,1),(8,1,-1)])
    
#(9,10,1),(9,2,1),(9,3,-1),(4,9,1),(5,9,-1),(10,4,0),(10,6,-1),(10,5,1),
#(11,12,1),(10,11,1),(9,11,-1),(11,9,1),(12,3,-1),(10,11,0),(11,12,-1),(10,7,1)

val_map = {1: 0.0,
           4: 1.0,
           5: 2.0,
           2: 1.0,
           8: 1.0}




values = [val_map.get(node, 0.25) for node in G.nodes()]


#print('weight for edge 1-3')
#print(G[1][3])


#number of nodes
size=G.number_of_nodes()



# Need to create a layout when doing
# separate calls to draw nodes and edges
pos = nx.spring_layout(G)
#orizei to layout gia thn optikopoihsh tou G grafou pou orisame parapanw
#pos:Initial positions for nodes as a dictionary with node as keys and values as a list or tuple.
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 500)
cmap=plt.get_cmap('jet')

nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_edges(G, pos, arrows=True)
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
Aminus = np.array(Am)
print('A-','\n',Aminus)
AminusTran = np.transpose(Aminus)
print('A- Tranpose','\n',AminusTran)

Bminus = Aminus.dot(AminusTran)
print('B-','\n',Bminus)        
Bplus = Aplus.dot(AplusTran)
print('B+','\n',Bplus)


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
            if edge_weight > 0:
                degree_map['out']['positive'] += 1
            else:
                degree_map['out']['negative'] += 1
        elif edge_dst == node:
            if edge_weight > 0:
                degree_map['in']['positive'] += 1
            else:
                degree_map['in']['negative'] += 1                
    
    #print('node:',node)
    #print('out positive',degree_map['out']['positive'])
    #print('in positive',degree_map['in']['positive'])
    #print('out negative',degree_map['out']['negative'])
    #print('in negative',degree_map['in']['negative'])        
    node_degrees.append(degree_map)

#print(node_degrees)
#print(node_degrees[7]['in']['negative'])





# create the Sums needed for the co-reference and co-citation matrices

def calculate_sigma1(i,j,k):
    s=0
    for c in range(k):
        s += np.sum(Aplus[i,k]*AplusTran[k,j])
    return s

def calculate_sigma2(i,j,k):
    s=0
    for c in range(k):
        s += np.sum(Aminus[i,k]*AminusTran[k,j])
    return s

def calculate_sigma3(i,j,k):
    s=0
    for c in range(k):
        s += np.sum(AplusTran[k,j]*Aplus[i,k])
    return s

def calculate_sigma4(i,j,k):
    s=0
    for c in range(k):
        s += np.sum(AminusTran[k,j]*Aminus[i,k])
    return s


#co-citation and co-reference matrices

Bp = [[0 for i in range(size)] for j in range(size)]

for i in range(size-1):
    for j in range(size-1):
        denominator = (node_degrees[i]['out']['positive'])*(node_degrees[j]['out']['positive'])
        if denominator == 0:
            Bp[i][j] = 0
        else :
            sigma = calculate_sigma1(i,j,size-1)
            Bp[i][j] = (1/(denominator))*sigma

print('\n','Bp','\n')
print(np.squeeze(np.asarray(Bp)))


Bm = [[0 for i in range(size)] for j in range(size)]

for i in range(size-1):
    for j in range(size-1):
        denominator = (node_degrees[i]['out']['negative'])*(node_degrees[j]['out']['negative'])
        if denominator == 0:
            Bm[i][j] = 0
        else :
            sigma = calculate_sigma2(i,j,size-1)
            Bm[i][j] = (1/(denominator))*sigma

print('\n','Bm','\n')
print(np.squeeze(np.asarray(Bm)))


Cp = [[0 for i in range(size)] for j in range(size)]

for i in range(size-1):
    for j in range(size-1):
        denominator = (node_degrees[i]['in']['positive'])*(node_degrees[j]['in']['positive'])
        if denominator == 0:
            Cp[i][j] = 0
        else :
            sigma = calculate_sigma3(i,j,size-1)
            Cp[i][j] = (1/(denominator))*sigma

print('\n','Cp','\n')
print(np.squeeze(np.asarray(Cp)))


Cm = [[0 for i in range(size)] for j in range(size)]

for i in range(size-1):
    for j in range(size-1):
        denominator = (node_degrees[i]['in']['negative'])*(node_degrees[j]['in']['negative'])
        if denominator == 0:
            Cp[i][j] = 0
        else :
            sigma = calculate_sigma4(i,j,size-1)
            Cp[i][j] = (1/(denominator))*sigma

print('\n','Cm','\n')
print(np.squeeze(np.asarray(Cm)))


#similarity and balance to be used as metrics for the affinity propagation algorithm


balance_in=[[0 for i in range(size)] for j in range(size)]
balance_out=[[0 for i in range(size)] for j in range(size)]
for i in range(size):
    for j in range(size):
        balance_in[i][j] = min(((1+Bp[i][j])/(1+Bm[i][j])),((1+Bm[i][j])/(1+Bp[i][j])))
        balance_out[i][j] = min(((1+Cp[i][j])/(1+Cm[i][j])),((1+Cm[i][j])/(1+Cp[i][j])))

print('ballance in','\n',np.squeeze(np.asarray(balance_in)))
print('balance out','\n',np.squeeze(np.asarray(balance_out)))


Sout=[[0 for i in range(size)] for j in range(size)]
Sin=[[0 for i in range(size)] for j in range(size)]
for i in range(size):
    for j in range(size):
        Sout[i][j]=balance_out[i][j]*(Bp[i][j] + Bm[i][j])
        Sin[i][j]=balance_in[i][j]*(Cp[i][j] + Cm[i][j])

print('\n','similarity out','\n')
print(np.squeeze(np.asarray(Sout)))
print('\n','similarity in','\n')
print(np.squeeze(np.asarray(Sin)))

similarity=[[0 for i in range(size)] for j in range(size)]
for i in range(size):
    for j in range(size):
        similarity[i][j]=Sin[i][j]+Sout[i][j]

print('\n','similarity','\n')

S=np.squeeze(np.asarray(similarity))
print(S)




'''
#normalisation of similarity matrix
mylist=[]
for i in range(size):
    mylist.append(max(S[i]))
print('mylist:',mylist)
print(max(mylist))

for i in range(size):
    for j in range(size):
        S[i][j] = (S[i][j]/(7))
print('\n',S)
'''
    
        
    

from sklearn.cluster import AffinityPropagation

# compute the similarity matrix
af = AffinityPropagation(affinity='precomputed', verbose=True , random_state=None)
af.fit(similarity)
print(af.labels_, af.cluster_centers_indices_)
number_of_clusters = len(af.cluster_centers_indices_)
print('number of clusters:' , number_of_clusters)


clusters=[]
for i in range(number_of_clusters):
    clusters.append([])
    

for i in range(size):
    for j in range(number_of_clusters):
        if  af.labels_[i] == j:
            clusters[j].append(i+1)
print(clusters)
print('\n')



#kmeans
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(similarity)
print('kmeans labels:',kmeans.labels_)
#kmeans.predict([[0, 0], [12, 3]])
#print(kmeans.cluster_centers_)
