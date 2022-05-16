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
    num_nodes = graph.number_of_nodes()
    adj_matrix = createAdjMatrix(graph,num_nodes)
    
    # Use the adjacency matrix to calculate the closeness centrality
    # https://medium.com/@pasdan/closeness-centrality-via-networkx-is-taking-too-long-1a58e648f5ce
    # Need to write our own algorithm
    total = 0.0
    
    # A big enough number to represent infinity
    INF = sys.maxsize
    
    closeness_centrality = {}
    for i in range(0, num_nodes):
        closeness_value = 0.0
        possible_paths = list(enumerate(adj_matrix[i:]))

        # shortest_paths = dict(filter( \
        # lambda x: not x[1] == INF, possible_paths))

        
        # print(shortest_paths.values())
        for values in len(possible_paths):
            for value in values:
                if value == 1:
                    total += 1
        
        #total += sum(shortest_paths.values())
        # print(total)
        n_shortest_paths = len(possible_paths) - 1.0
        if total > 0.0 and num_nodes > 1:
            s = n_shortest_paths / (num_nodes - 1)
            closeness_value = (n_shortest_paths / total) * s
        closeness_centrality[i]=closeness_value


        # for j in range(0, num_nodes):
        #     if adj_matrix[i][j] != 0:
        #         closeness_value += 1.0
        # adj_matrix[i][i] = closeness_value


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
                if graph.has_edge(G_nodes[i],G_nodes[j]):
                    adj_matrix[i][j] = 1
                else:
                    adj_matrix[i][j] = INF
            #print(adj_matrix[i][j])
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
    
    
'''
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
                    adj_matrix[i][j] = 999
    return adj_matrix

    #Create floyd warshall algorithm



if rank ==0:




'''
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
closeness = closeness_centrality(G_und)
print(len(G.nodes))
print(G)

# Calculate closeness centrality using NetworkX for testing and compare with Floyd-Warshall Algorithm
closeness = closeness_centrality(G_und)

# Write closeness centrality into output.txt
with open('output.txt', 'w') as f:
    for key in closeness:
        f.write(str(key) + ' ' + str(closeness[key]) + '\n')

# Print five nodes with the top centrality values (if there are more than five nodes with the same centrality values,
# then print any five nodes with those values) and also the average of the centrality values of all nodes on screen
top_five = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
print("The top 5 nodes are: "+ str(top_five))
print("The average of the closeness centrality values of all nodes is: "+ str(average(list(closeness.values()))))

# Make a graph for closeness centrality
plt.hist(closeness.values(), bins=100)
plt.title('Closeness Centrality')
plt.show()