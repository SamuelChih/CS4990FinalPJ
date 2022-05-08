import gzip
from numpy import average
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


raw = gzip.open('twitter_combined_reduced.txt.gz')

# Read in dataset using NetworkX
G = nx.read_edgelist(raw, create_using=nx.DiGraph(), nodetype=int)
G_und = G.to_undirected()
closeness = closeness_centrality(G_und)
print(len(G.nodes))
print(G)

# Calculate closeness centrality using NetworkX for testing and compare with Floyd-Warshall Algorithm
closeness = closeness_centrality(G_und)