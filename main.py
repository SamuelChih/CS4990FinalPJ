import gzip
from numpy import average
import networkx as nx
import matplotlib.pyplot as plt
from platform import processor
from mpi4py import MPI
import sys


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




# def createAdjMatrix(graph, num_nodes):
#     size = num_nodes
#     AdjMatrix = [[]*size]*size
#     G_nodes = list(graph.nodes())

    

#     for i in range(size):
#         for j in range(size):
#             # if(i == j):
#             #     AdjMatrix[i][j] = 0
#             if(G.has_edge(G_nodes[i],G_nodes[j])):
#                 AdjMatrix[i][j] = 1

#             # print(AdjMatrix[i][j])
    
#     return AdjMatrix
    
    

# #######################
#   Floyds Algo Pseudo
#   Floyds All Pairs Shortsest 
  
#   procedure FWSP(A)
  
#   begin
#       D^(0) = A
#       for k = 1 to n do:
#           for i = 1 to n do: # <= Broadcast nth row ot all processors
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
closeness = closeness_centrality(G_und)
print(len(G.nodes))
print(G)

# Calculate closeness centrality using NetworkX for testing and compare with Floyd-Warshall Algorithm
closeness = closeness_centrality(G_und)

# # Write closeness centrality into output.txt
# with open('output.txt', 'w') as f:
#     for key in closeness:
#         f.write(str(key) + ' ' + str(closeness[key]) + '\n')

# Print five nodes with the top centrality values (if there are more than five nodes with the centrality values, then print any five) 
# and also the average of the centrality values of all nodes on screen.
top_five = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
print("The top 5 nodes are: "+ str(top_five))
print("The average of the closeness centrality values of all nodes is: "+ str(average(list(closeness.values()))))

# Make a graph for closeness centrality
plt.hist(closeness.values(), bins=100)
plt.title('Closeness Centrality')
plt.show()