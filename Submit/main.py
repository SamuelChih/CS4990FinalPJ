import gzip
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()


# Calculate closeness centrality using Floyd-Warshall Algorithm, using tge heNetworkX
# def closeness_centrality(graph):
    
#     # Initialize the adjacency matrix
    
#     num_nodes = graph.number_of_nodes()

#     start = timeit.default_timer()
#     adj_matrix = create_adjacency_matrix_mpi(graph)
#     stop = timeit.default_timer()
#     print("Time: "+str(stop-start))

#     path_length = nx.single_source_shortest_path_length
#     closeness_centrality = {}

#     nodes = graph.nodes
#     closeness_centrality = {}

#     # Adjacency metrix to closeness centrality
#     # Refrence: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
#     for n in nodes:
#         sp = path_length(graph, n)
#         totsp = sum(sp.values())
#         len_G = len(adj_matrix)
#         _closeness_centrality = 0.0
#         if totsp > 0.0 and len_G > 1:
#             _closeness_centrality = (len(sp) - 1.0) / totsp
#             # normalize to number of nodes-1 in connected part
#             s = (len(sp) - 1.0) / (len_G - 1)
#             _closeness_centrality *= s
#         closeness_centrality[n] = _closeness_centrality

#     return closeness_centrality


def closeness_centrality(graph):
    # Initialize the adjacency matrix
    num_nodes = len(graph)
    npa = nx.to_numpy_array(graph)
    #Convert graph to adjacency matrix
    # adj_matrix = create_adjacency_matrix(G, len(G))

    print(len(G.nodes))
    print(G)
    newG = [[]]

    for i in range(len(npa)):
        for j in range(len(npa)):
            if npa[i][j] == 0:
                npa[i][j] = 999

    adj_matrix = create_adjacency_matrix_mpi(npa)

    print("the adj matrix: " + str(adj_matrix))
    total = 0.0

    # A big enough number to represent infinity
    INF = 999
    
    closeness_centrality = {}
    for i in range(0, num_nodes):
        closeness_value = 0.0
        possible_paths = list(enumerate(adj_matrix[i :]))
        
        # Look for shortest paths of length 1 and add each occurrence to total
        #print(len(possible_paths))
        
        #total = np.sum(possible_paths.values()) - possible_paths['999']
        for k in range(len(possible_paths)):
            for j in range(len(possible_paths)):
                if possible_paths[k][1][j] == 1:
                    total += 1
        #print("Total", total)
        
        # print("possible paths: "+str(possible_paths))
        # shortest_paths = dict(filter( \
        # lambda x: not x[1] == INF, possible_paths))

        #filter number that is not infinity
        #shortest_paths = dict(filter(lambda x: x[1] != INF, possible_paths))
        # print("==========================SEEE MEE=============================")
        
        # shortest_paths = filter(lambda x: x[1] != 999, possible_paths)
        # print("SHORTTTTTT",shortest_paths)

        # print("SHORTEST PATHS",str(shortest_paths))
        
       
        # print(shortest_paths.values())
        # for values in shortest_paths.values():
        #     for value in values:
        #         if value == 1:
        #             total += 1
        
        #total += sum(shortest_paths.values())
        # print(total)
        total = total /187
        n_shortest_paths = total - 1.0
        
        if total > 0.0 and num_nodes > 1:
            
            s = n_shortest_paths / (num_nodes - 1)
            # print("n_shortest_paths: ", n_shortest_paths)
            # print("num_nodes: ", num_nodes)
            # print("s: ", s)
            closeness_value = (n_shortest_paths / total) * s
            # print("closeness_value: ", closeness_value)
        total = 0
        closeness_centrality[i] = closeness_value #i should be user ID

    return closeness_centrality



# Implement MPI parallelization

# Create an adjacency matrix for the graph serially
def create_adjacency_matrix(graph, num_nodes):
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
            # print(adj_matrix[i][j])
    return adj_matrix


# Create an adjacency matrix for the graph using MPI
def create_adjacency_matrix_mpi(graph):
    n = len(graph)
    matrix_slice = 0
    excess = 0
    
    # Rank 0 sends slices of the adjacency matrix to all other processes
    if rank == 0:
        finish = 0
        result = []
        comm.bcast(n, root = 0) # Broadcast the number of nodes

        excess = n % (size - 1)
        matrix_slice = (n - excess) / (size - 1)
        
        #matrix_slice = 1 #Uncomment this line to get the MPI matrix
        
        # Send slices
        for i in range(1, size):
            comm.send(graph, i, finish)
            while (True):
                comm.recv(result, 3, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                if status.Get_tag() == finish:
                    finish += 1
                else:
                    if graph[(result[1] * n) + result[2]] > result[0]:
                        graph[(result[1] * n) + result[2]] = result[0]
                if finish >= size - 1:
                    break
      
      # Set up message to send
            msg = []
            comm.recv(n, 1, 0, status)
            comm.recv(graph, n * n, 0, status)
            
            if rank + 1 != size:
                excess = 0
                        
            for k in range(matrix_slice * (rank - 1), matrix_slice * (rank - 1) + matrix_slice + excess):
                for i in range(len(n)):
                    for j in range(len(n)):
                        if graph[(i * n) + k] * graph[(k * n) + j] != 0 and i != j:
                            graph[(i * n) + j] = graph[(i * n) + k] + graph[(k * n) + j]
                            msg[0].append(graph[(i * n) + j])
                            msg[1].append(i)
                            msg[2].append(j)
                            comm.send(msg, 3, 0)
    #print(graph)
    return graph


# Create an adjacency matrix for the graph with MPI
# def create_adjacency_matrix_mpi(graph):
#     # Size of adjacency matrix
#     n = len(graph)
    
#     # Size of adjacency matrix / # of processors
#     mag = n / size
    
#     start = mag * rank
#     end = (mag * (rank + 1))
#     for k in range(1, n + 1):
#         owner = int((size / n) * (k - 1))
#         graph[k-1] = comm.bcast(graph[k - 1], root = owner)
#         for i in range(start, end):
#             if ((i + 1) != k):
#                 for j in range(0, n):
#                     if ((j + 1) != k):
#                         graph[i][j] = min(graph[i][j], graph[i][k - 1] + graph[k - 1][j])

#     for k in range(1, n + 1):
#         owner = int((size / n) * (k - 1))
#         graph[k - 1] = comm.bcast(graph[k - 1], root = owner)
#     return graph
    



# #######################
#   Floyds Algo Pseudo
#   Floyds All Pairs Shortest 
  
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
#raw = gzip.open('facebook_combined.txt.gz')

# Test dataset
raw = gzip.open('twitter_combined_reduced.txt.gz')

# Read in dataset using NetworkX
G = nx.read_edgelist(raw, create_using=nx.DiGraph(), nodetype=int)

G_und = G.to_undirected()


# Calculate closeness centrality using Floyd-Warshall Algorithm
closeness = closeness_centrality(G_und)

# Write closeness centrality into output.txt
with open('output.txt', 'w') as f:
    for key in closeness:
        f.write(str(key) + ' ' + str(closeness[key]) + '\n')

# Print five nodes with the top centrality values (if there are more than five nodes with the same centrality values,
# then print any five nodes with those values) and the average of the centrality values of all nodes on screen

top_five = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
print("The top 5 nodes are: "+ str(top_five))
print("The average of the closeness centrality values of all nodes is: "+ str(np.average(list(closeness.values()))))

# Make a histogram for closeness centrality
# plt.hist(closeness.values(), bins=100)
# plt.title('Closeness Centrality')
# plt.show()