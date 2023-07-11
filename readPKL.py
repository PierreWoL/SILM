import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,cut_tree
from clustering import AgglomerativeClustering_param_search
import pygraphviz
"""
# Generate random data
np.random.seed(0)
X = np.random.randn(10, 2)  # Replace with your own data
print(X)
# Perform hierarchical clustering
Z = linkage(X, method='complete')  # You can choose different linkage methods

# Plot dendrogram
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')

dendrogram(Z)

plt.show()
# Specify the number of desired clusters
num_clusters = 3  # Adjust as per your requirement

# Cut the hierarchical tree to obtain clusters
clusters = cut_tree(Z, n_clusters=num_clusters).flatten()
print(clusters)
# Output the clusters and the data points within each cluster
unique_clusters = np.unique(clusters)

for cluster in unique_clusters:
    cluster_indices = np.where(clusters == cluster)[0]
    cluster_points = X[cluster_indices]
    print(f"Cluster {cluster}:")
    print(cluster_indices)
def extract_hierarchy(Z):
    hierarchy = {}
    for i, linkage in enumerate(Z):
        cluster = {}
        cluster["id"] = i
        cluster["left"] = int(linkage[0])
        cluster["right"] = int(linkage[1])
        cluster["distance"] = linkage[2]
        cluster["size"] = linkage[3]
        hierarchy[i] = cluster
    return hierarchy


# Output hierarchy
hierarchy = extract_hierarchy(Z)
for cluster in hierarchy.values():
    print(cluster)


"""


import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph object
tree = nx.DiGraph()

# Add nodes and edges to the graph
tree.add_edge("A", "B")
tree.add_edge("A", "C")
tree.add_edge("B", "D")
tree.add_edge("B", "E")

# Define the layout for the tree structure with top-down direction
pos = nx.nx_agraph.graphviz_layout(tree, prog='dot', args="-Grankdir=TB")

# Draw the tree structure
plt.figure(figsize=(8, 6))
nx.draw(tree, pos, with_labels=True, node_color='lightblue', node_size=1500, arrows=True)
plt.title("Tree Structure")
plt.show()