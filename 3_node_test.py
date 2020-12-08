# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:38:19 2020

@author: Manos
"""

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
from sklearn.cluster import AffinityPropagation




#create the graph
G = nx.DiGraph()
G.add_weighted_edges_from(
    [(1,3,1),(1,2,-1),(3,1,-1),(2,1,1),(2,3,-1),(3,2,1)])
    
val_map = {1: 0.0,
           4: 1.0,
           5: 2.0,
           2: 1.0,
           8: 1.0}

values = [val_map.get(node, 0.25) for node in G.nodes()]


#number of nodes
size=G.number_of_nodes()


pos = nx.spring_layout(G)
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


#adjacency matrix
converted = nx.convert_matrix.to_numpy_matrix(G)
#print(converted)





#creating matrices
c = copy.deepcopy(converted)
Ap = np.squeeze(np.asarray(c))
Ap[Ap < 0] = 0
#print('Ap','\n',Ap)

c = copy.deepcopy(converted)
Am = np.squeeze(np.asarray(c))
Am[Am > 0] = 0
#print('Am','\n',Am)





#define Aplus , Aplustransponse etc as numpy arrays
Aplus = np.array(Ap)
#print('A+','\n',Aplus)
AplusTran = np.transpose(Aplus)
#print('A+ Tranpose','\n',AplusTran)
Aminus = np.array(Am)
#print('A-','\n',Aminus)
AminusTran = np.transpose(Aminus)
#print('A- Tranpose','\n',AminusTran)




#create degree map

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
            if int(edge_weight) > 0:
                degree_map['out']['positive'] += 1
            else:
                degree_map['out']['negative'] += 1
        elif edge_dst == node:
            if int(edge_weight) > 0:
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
#print(node_degrees[3]['in']['negative'])





# create the Sums needed for the co-reference and co-citation matrices

def calculate_sigma1(i, j, num_nodes):
    s = 0
    for k in range(num_nodes):
        s += Aplus[i,k]*AplusTran[k,j]
    return s

def calculate_sigma2(i, j, num_nodes):
    s=0
    for k in range(num_nodes):
        s += Aminus[i,k]*AminusTran[k,j]
    return s

def calculate_sigma3(i,j,num_nodes):
    s=0
    for k in range(num_nodes):
        s += AplusTran[k,j]*Aplus[i,k]
    return s

def calculate_sigma4(i,j,num_nodes):
    s=0
    for k in range(num_nodes):
        s += AminusTran[k,j]*Aminus[i,k]
    return s


# Co-citation and co-reference matrices

##############################################################
Bp = np.zeros([size, size])
for i in range(size):
    for j in range(size):
        denominator = (node_degrees[i]['out']['positive'] * node_degrees[j]['out']['positive'])
        if denominator == 0:
            Bp[i,j] = 0
        else:
            sigma = calculate_sigma1(i,j,size)
            Bp[i,j] = (1 / denominator) * sigma


###########################################################
Bm = np.zeros([size, size])
for i in range(size):
    for j in range(size):
        denominator = (node_degrees[i]['out']['negative'] * node_degrees[j]['out']['negative'])
        if denominator == 0:
            Bm[i,j] = 0
        else:
            sigma = calculate_sigma2(i, j, size)
            Bm[i,j] = (1 / denominator) * sigma


######################################################################
Cp = np.zeros([size, size])
for i in range(size):
    for j in range(size):
        denominator = (node_degrees[i]['in']['positive'] * node_degrees[j]['in']['positive'])
        if denominator == 0:
            Cp[i,j] = 0
        else :
            sigma = calculate_sigma3(i, j, size)
            Cp[i,j] = (1 / denominator) * sigma


###################################################################################
Cm = np.zeros([size, size])
for i in range(size):
    for j in range(size):
        denominator = (node_degrees[i]['in']['negative'] * node_degrees[j]['in']['negative'])
        if denominator == 0:
            Cp[i,j] = 0
        else :
            sigma = calculate_sigma4(i, j, size)
            Cp[i,j] = (1 / denominator)*sigma





# Balance
balance_in = np.zeros([size, size])
balance_out = np.zeros([size, size])
for i in range(size):
    for j in range(size):
        
        fraction1 = (1 + Bp[i,j]) / (1 + Bm[i,j])
        fraction2 = (1 + Bm[i,j]) / (1 + Bp[i,j])
        balance_in[i,j] = min(fraction1, fraction2)
        
        fraction3 = (1 + Cp[i,j]) / (1+Cm[i,j])
        fraction4 = (1 + Cm[i,j]) / (1+Cp[i,j])
        balance_out[i,j] = min(fraction3, fraction4)





# Similarity
#######################################
Sout= np.zeros([size, size])
Sin= np.zeros([size, size])
for i in range(size):
    for j in range(size):
        Sout[i,j]=balance_out[i,j] * ( Bp[i,j] + Bm[i,j] )
        Sin[i,j]=balance_in[i,j] * ( Cp[i,j] + Cm[i,j] )


similarity =  [[0 for i in range(size)] for j in range(size)]
for i in range(size):
    for j in range(size):
        similarity[i][j] = Sin[i,j] + Sout[i,j]





#############################################
#PRINT
#############################################
print('\n','Cp','\n', Cp)
print('\n','Cm','\n', Cm)
print('\n','Bp','\n', Bp)
print('\n','Bm','\n', Bm)

print('\n','balance in','\n', balance_in)
print('\n','balance out','\n', balance_out)

print('\n','similarity in','\n', Sin)        
print('\n','similarity out','\n', Sout)

print('\n','similarity','\n', similarity)


    
# run the affinity propagation algorithm
af = AffinityPropagation(affinity='precomputed', verbose=True , random_state=0)
af.fit(similarity)
#print(af.labels_, af.cluster_centers_indices_)
number_of_clusters = len(af.cluster_centers_indices_)
clusters=[]
for i in range(number_of_clusters):
    clusters.append([])
    
for i in range(size):
    for j in range(number_of_clusters):
        if  af.labels_[i] == j:
            clusters[j].append(i+1)


print('\n','Number of clusters:' , number_of_clusters)
print('\n','Cluster centers:',af.cluster_centers_indices_)
print('\n','Clusters:')
print(clusters)
print('\n')

#print('degrees:', degrees)
print('number of edges:', G.number_of_edges())
print('number of nodes:', G.number_of_nodes())



print(node_degrees)

'''
#kmeans
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=5, random_state=0).fit(similarity)
print('\n','kmeans labels:',kmeans.labels_)
print('kmeans cluster centers:', kmeans.cluster_centers_)
'''