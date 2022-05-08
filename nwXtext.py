import gzip
from statistics import mean, median, stdev
import networkx as nx
import matplotlib.pyplot as plt
from platform import processor
from mpi4py import MPI

#raw = gzip.open('facebook_combined.txt.gz')
raw = gzip.open('twitter_combined.txt.gz')
#raw = gzip.open('twitter_combined_reduced.txt.gz')
# Read in dataset using NetworkX
G = nx.read_edgelist(raw, create_using=nx.DiGraph(), nodetype=int)
G_und = G.to_undirected()
print(G)

# Calculate closeness centrality using NetworkX for testing
closeness = nx.closeness_centrality(G_und)
#1m 7.8s for facebook_combined.txt.gz
#0.2s for twitter_combined_reduced.txt.gz

#write betwness in to file output.txt
with open('output_nwx_twitter.txt', 'w') as f:
    for key in closeness:
        f.write(str(key) + ' ' + str(closeness[key]) + '\n')
        
# #write betwness in to file output.txt
# with open('output_nwx_fb.txt', 'w') as f:
#     for key in closeness:
#         f.write(str(key) + ' ' + str(closeness[key]) + '\n')

# Make a graph for closeness centrality
plt.hist(closeness.values(), bins=100)
plt.title('Closeness Centrality')
plt.show()