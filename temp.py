import gzip
from numpy import append, average
import networkx as nx
import matplotlib.pyplot as plt
from platform import processor
from mpi4py import MPI
import sys

from numpy import average

# Calculate closeness centrality using Floyd-Warshall Algorithm
def closeness_centrality(graph):
    
    # Initialize the adjacency matrix
    num_nodes = graph.number_of_nodes()
    closeness_graph = createAdjMatrix(graph,num_nodes)
    return closeness_graph
        
# Create an adjacency matrix for the graph
def createAdjMatrix(graph,num_nodes):
    # A big enough number to represent infinity
    INF = 999
    
    # Initialize the adjacency matrix
    G_nodes = list(graph.nodes())
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i][j] = 0
            else:
                if graph.has_edge(G_nodes[i],G_nodes[j]):
                    adj_matrix[i][j] = 1
                else:
                    adj_matrix[i][j] = INF
            print(adj_matrix[i][j])
    return adj_matrix


#Calculate Floyd-Warshall Algorithm for the adjacency matrix using MPI parallelization
# def floyd_warshall(adj_matrix,num_nodes):
#     # A big enough number to represent infinity
#     INF = 999
#     # Initialize the adjacency matrix
#     G_nodes = list(graph.nodes())
#     # Initialize the distance matrix
#     dist_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)] # Initialize the distance matrix
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             dist_matrix[i][j] = adj_matrix[i][j]
#     # Initialize the predecessor matrix
#     pred_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)] # predecessor matrix
#     for i in range(num_nodes):  # initialize predecessor matrix
#         for j in range(num_nodes):
#             pred_matrix[i][j] = i   # initialize predecessor matrix
#     # Initialize the shortest path matrix
#     path_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)] # shortest path matrix
#     for i in range(num_nodes):  # initialize shortest path matrix   # initialize shortest path matrix
#         for j in range(num_nodes):  # initialize shortest path matrix
#             path_matrix[i][j] = []  # initialize shortest path matrix   
#     # Initialize the shortest path matrix
#     path_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)] # shortest path matrix  
#     for i in range(num_nodes):  # initialize shortest path matrix
#         for j in range(num_nodes):  # initialize shortest path matrix
#             path_matrix[i][j] = []  # initialize shortest path matrix
#     # Initialize the shortest path matrix 
    

# #Calculate Floyd-Warshall Algorithm for the adjacency matrix
# def floyd_warshall(adj_matrix):
#     # A big enough number to represent infinity
#     INF = 999
#     # Initialize the adjacency matrix
#     num_nodes = len(adj_matrix)
#     for k in range(num_nodes):  # k is the source node
#         for i in range(num_nodes): # i is the destination node
#             for j in range(num_nodes): # j is the intermediate node
#                 if adj_matrix[i][j] > adj_matrix[i][k] + adj_matrix[k][j]:
#                     adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j]
#     return adj_matrix

raw = gzip.open('twitter_combined_reduced.txt.gz')

# Read in dataset using NetworkX
G = nx.read_edgelist(raw, create_using=nx.DiGraph(), nodetype=int)
G_und = G.to_undirected()
closeness = closeness_centrality(G_und)
print(len(G.nodes))
print(G)

# Calculate closeness centrality using NetworkX for testing and compare with Floyd-Warshall Algorithm
closeness = closeness_centrality(G_und)