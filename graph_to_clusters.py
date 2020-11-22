# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:54:45 2020

@author: Manos
"""


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from functools import reduce
import copy
import pandas as pd





def create_graph_from_file(filepath):
    '''reads txt file and creates graph'''
    G = nx.DiGraph()
    with open(filepath ,  encoding="utf8") as f:
       line = f.readline()
       cntr = 1
       while line:
           #print(line.strip())
           cntr += 1
           if line.startswith('SRC') :
               #print(line.strip())
               src = line.strip().split('SRC:',1)[1]
               #print(src)
               line_tgt = f.readline()
               tgt = line_tgt.strip().split('TGT:',1)[1]
               #print(tgt)
               line_vot = f.readline()
               vot = line_vot.strip().split('VOT:',1)[1]
               #print(vot)
               G.add_weighted_edges_from([(src,tgt,vot)])
           if cntr == 1000 :
               return G
           line = f.readline()
       #return G
           
def visualize_graph(g):
    '''visualize the graph'''
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size = 500)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos, arrows=True)
    plt.show()
   
    
G = create_graph_from_file('C:\\Users\Manos\Desktop\wiki-RfA.txt')
print(G.number_of_edges())
print(G.number_of_nodes())
#visualize_graph(G)

size=G.number_of_nodes()

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




'''
import seaborn as sns
sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation

S, clusters = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(S[:,0], S[:,1], alpha=0.7, edgecolors='b')

af = AffinityPropagation(preference=-50)
clustering = af.fit(S)

plt.scatter(S[:,0], S[:,1], c=clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
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







