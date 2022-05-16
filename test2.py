import gzip
from numpy import append, average
import networkx as nx
import matplotlib.pyplot as plt
from platform import processor
from mpi4py import MPI
import sys


# Calculate closeness centrality using Floyd-Warshall Algorithm
def closeness_centrality(graph):
    
    # Initialize the adjacency matrix
    nodes = graph.number_of_nodes()
    adj_matrix = createAdjMatrix(graph,nodes)

    path_length = nx.single_source_shortest_path_length
    
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(adj_matrix)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            s = (len(sp) - 1.0) / (len_G - 1)
            _closeness_centrality *= s
        closeness_centrality[n] = _closeness_centrality

    return closeness_centrality


# Implement MPI parallelization

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
                if graph.has_edge(G_nodes[i], G_nodes[j]):
                    adj_matrix[i][j] = 1
                else:
                    adj_matrix[i][j] = INF
            #print(adj_matrix[i][j])
    return adj_matrix

# #######################
#   Floyds Algo Pseudo
#   Floyds All Pairs Shortsest 
  
#   procedure FWSP(A)
  
#   begin
#       D^(0) = A
#       for k = 1 to n do:
#           for i = 1 to n do: # <= Broadcast kth row to all processors using MPI
#               for j = 1 to n do:
#                   d(k)_i,j = min(d(k-1)_i,j,d(k-1)i,k+d(k-1)_k,j)
#   end FWSP
    
# ######################


# Full dataset
# raw = gzip.open('facebook_combined.txt.gz')

# Test dataset
raw = gzip.open('twitter_combined_reduced.txt.gz')

# Read in dataset using NetworkX
G = nx.read_edgelist(raw, create_using=nx.DiGraph(), nodetype=int)
G_und = G.to_undirected()
print(len(G.nodes))

# Calculate closeness centrality using NetworkX for testing and compare with Floyd-Warshall Algorithm
closeness = closeness_centrality(G_und)

# Write closeness centrality into output.txt


# with open('output.txt', 'w') as f:
#     for key in closeness:
#         f.write(str(key) + ' ' + str(closeness[key]) + '\n')

# Print five nodes with the top centrality values (if there are more than five nodes with the same centrality values,
# then print any five nodes with those values) and also the average of the centrality values of all nodes on screen
top_five = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
print("The top 5 nodes are: "+ str(top_five))
print("The average of the closeness centrality values of all nodes is: "+ str(average(list(closeness.values()))))

# Make a histogram for closeness centrality
# plt.hist(closeness.values(), bins=100)
# plt.title('Closeness Centrality')
# plt.show()