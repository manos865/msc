# Adjacency Matrix representation in Python

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Graph(object):
    
    #self represents the instances of the class
    #class: o grafos
    #instances : ta stoixeia tou grafou: akmes kai korufes
    # Initialize the matrix
    #function , init declares the instance of class 
    #self is the particular instance
    #arxika ftiaxnoume kenous pinakes
    # the graph is a class that encompasses several objects, the vertices and edges
    def __init__(self, size):
        #empty adjacency matrix
        self.adjMatrix = []
        self.Ap = []
        self.Am = []
        #
        #add the value 0 to the matrix
        #ka8orizoume to mege8os tou pinaka
        #otan trexei prepei na exei kati 
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
            self.Ap.append([0 for i in range(size)])
            self.Am.append([0 for i in range(size)])
            self.size = size
        
    # Add edges
    def add_edge(self, v1, v2):
        #if statement in order to not add the same edge twice
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
        #add the proper values to the adjacency matrix
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = -1
        self.Ap[v1][v2] = 1
        self.Ap[v2][v1] = 0
        self.Am[v1][v2] = 0
        self.Am[v2][v1] = -1

    # Remove edges
    def remove_edge(self, v1, v2):
        # check that the value of the adjacency matrix is 0
        # for every pair of vertices where there is no edge
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        # den sundeontai ara vazoume mhdenikh timh
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0
        self.Ap[v1][v2] = 0
        self.Ap[v2][v1] = 0
        self.Am[v1][v2] = 0
        self.Am[v2][v1] = 0
        
    #self : korufes (vertices/nodes), akmes (edges)
    #print the size of the matrix
    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):

        
        #self.adjMatrix lista apo listes
        '''
        A=('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
                         for row in self.adjMatrix]))
        Aplus=('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
                         for row in self.Ap]))
        Aminus=('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
                         for row in self.Am]))
        print('\n', A , '\n')
        print('\n', Aminus , '\n')
        print('\n', Aplus , '\n')
            '''        
        
        
        Adj=np.array(self.adjMatrix)
        print(Adj)
        
        Aplus=np.array(self.Ap)
        print('A+','\n',Aplus)
        AplusTran=np.transpose(Aplus)
        print('A+ Tranpose','\n',AplusTran)
        Bplus=Aplus.dot(AplusTran)
        print('B+','\n',Bplus)
        
        Aminus=np.array(self.Am)
        print('A-','\n',Aminus)
        AminusTran=np.transpose(Aminus)
        print('A- Tranpose','\n',AminusTran)
        Bminus=Aminus.dot(AminusTran)
        print('B-','\n',Bminus)        
        
        Cminus=AminusTran.dot(Aminus)
        print('C-','\n',Cminus)
        
        Cplus=AplusTran.dot(Aplus)
        print('C+','\n',Cplus)
        
        
        
        #Dop positive outdegree
        #Dom negative outdegree
        #Dip positive indegree
        #Dim negative indegree
        
        '''
        for i in range(size):
            for j in range(size) :
                for k in range(size):
                    Bp=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
                    Bm=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
                    Cp=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
                    Cm=(1/Dop(i)*Dom(j))*   Aplus[i,k]*AplusTran[k,j]
        
        
        G = nx.DiGraph()
        nx.add_path(G, [0, 1, 2, 3])
        G.out_degree(0)  # node 0 with degree 1
        list(G.out_degree([0, 1, 2]))
        '''
        
#insert the variables for the graph 
def main():
    g = Graph(5)
    for i in range (0,5):
        for j in range (0,5):
            if i != j :
                g.add_edge(i, j)
    #g.add_edge(0, 1)
    #g.add_edge(0, 2)
    #g.add_edge(1, 2)
    #g.add_edge(2, 0)
    #g.add_edge(2, 3)
    print(g)
    g.print_matrix()
    
   #print('\n','degree:',g.in_degree[0])
    
    #if kai loop gia to add edge kai na vazeis osous sunduasmous

if __name__ == '__main__':
    main()



