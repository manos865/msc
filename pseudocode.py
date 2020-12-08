# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:01:06 2020

@author: Manos
"""



x = dict()
x['out'] = dict()
x['in'] = dict()
x['out']['positive'] = 3
x['out']['negative'] = 0
x['in']['positive'] = 6
x['in']['negative'] = 9
#x['in']['positive'] = 7


thelist=[]
for i in range(0,3):
    thelist.append(x)
print(thelist)


################################################
node_degrees = []
for node in G.nodes():
    degree_map = dict()
    degree_map['out'] = dict()
    degree_map['in'] = dict()
    
    degree_map['out']['positive'] = 0
    degree_map['out']['negative'] = 0
    degree_map['in']['positive'] = 0
    degree_map['in']['negative'] = 0
    #for edge in node.edges():
        #if edge out:
            #if edge.weight > 0 :
                #degree_map['out']['positive']+=1