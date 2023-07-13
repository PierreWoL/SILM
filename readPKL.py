import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score

"""

This is for a test case to construct a  directed graph object based
on ground truth hierarchy
"""


def hierarchy_tree(tree: nx.DiGraph()):
    # Define the layout for the tree structure with top-down direction
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot', args="-Grankdir=TB")
    # Draw the tree structure
    plt.figure(figsize=(8, 6))
    nx.draw(tree, pos, with_labels=True, node_color='lightblue', node_size=1500, arrows=True)  # pos,
    plt.title("Tree Structure")
    plt.show()


def present_children(tree, parent):
    # Print the related nodes (children)
    children = list(tree.successors(parent))
    print("Children:", children)


# Create a directed graph object
tree1 = nx.DiGraph()
# Add nodes and edges to the graph
tree1.add_edge("A", "B")
tree1.add_edge("A", "C")
tree1.add_edge("B", "D")
tree1.add_edge("B", "E")
hierarchy_tree(tree1)


# Sample distance matrix (or similarity matrix)


# Perform hierarchical clustering


def plot_tree(linkage_matrix, node_labels=None):
    plt.clf()
    dendrogra = sch.dendrogram(linkage_matrix, labels=node_labels)
    # Set plot labels and title
    plt.xlabel('Distance')
    plt.ylabel('Nouns')
    plt.title('Noun Dendrogram')
    # Display the plot
    plt.show()
    return dendrogra


def dendrogram_To_DirectedGraph(linkage_matrix, labels):
    # Create a new NetworkX tree object
    tree = nx.DiGraph()
    # Iterate over the linkage matrix
    for i, link in enumerate(linkage_matrix):
        child_1, child_2, distance, _ = link
        parent_node = i + len(encodings)  # Assign unique IDs to parent nodes dist_matrix
        # Add edges to the tree object
        tree.add_edge(parent_node, int(child_1))
        tree.add_edge(parent_node, int(child_2))
    # Assign labels to the tree nodes
    tree = nx.relabel_nodes(tree, {i: label for i, label in enumerate(labels)})
    hierarchy_tree(tree)
    return tree


# encodings = np.random.rand(10, 3)  # Assume each noun has a 5-dimensional encoding
encodings = np.array([[0.29661852, 0.95912228, 0.25205271],
                      [0.98564082, 0.37565498, 0.73105763],
                      [0.93450012, 0.12862963, 0.86722713],
                      [0.14059451, 0.73131089, 0.59513037],
                      [0.52330052, 0.45879583, 0.04439388],
                      [0.7673861, 0.73452903, 0.14095513],
                      [0.15567008, 0.95688681, 0.96494895],
                      [0.79879999, 0.56013694, 0.9268367],
                      [0.98665682, 0.55432251, 0.75036973],
                      [0.92581121, 0.68291704, 0.15144076]])
node_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
linkage_matrix = sch.linkage(encodings, method='complete', metric='euclidean')
dendrogra = plot_tree(linkage_matrix, node_labels=node_labels)
tree_test = dendrogram_To_DirectedGraph(linkage_matrix, node_labels)


# Convert the dendrogram to a tree structure
def dendrogram_to_tree(dn):
    children = {}
    last_i = None
    for i, d, c, _ in zip(dn['icoord'], dn['dcoord'], dn['color_list'], dn['ivl']):
        if i[0] not in children:
            children[i[0]] = {'parent': None, 'children': [], 'distance': None}
        if i[2] not in children:
            children[i[2]] = {'parent': None, 'children': [], 'distance': None}

        children[i[0]]['children'].append(i[2])
        children[i[2]]['parent'] = i[0]
        children[i[2]]['distance'] = d[1]
        last_i = i

    root = last_i[0]
    while children[root]['parent'] is not None:
        root = children[root]['parent']

    return {'root': root, 'children': children}


# Print the tree structure
def print_tree(tree, node, level=0):
    indent = '    ' * level
    print(f"{indent}Node: {node}")
    for child in tree['children'].get(node, {}).get('children', []):
        print_tree(child, level + 1)


"""tree = dendrogram_to_tree(dendrogra)
print_tree(tree['root'])"""
# Generate a random distance matrix


# Set the horizontal line to slice the dendrogram
threshold = 0.5
# Cut the dendrogram based on the threshold
clusters = sch.cut_tree(linkage_matrix, height=threshold)
print(clusters)
silhouette_avg = silhouette_score(encodings, clusters)
print(f"Silhouette coefficient: {silhouette_avg}")
# TODO: we need to iterate the threshold and test the silhouette score

"""
The following is for the inferr the cluster in the tree

"""
closest_parent = nx.lowest_common_ancestor(tree_test, 'A', 'B')

# Nodes A, B, and D
nodes = ['A', 'B', 'D']

# Find common ancestors of nodes A, B, and D
common_ancestors = set(tree_test.ancestors(nodes[0]))
for node in nodes[1:]:
    common_ancestors = common_ancestors.intersection(tree_test.ancestors(node))

print("Common Ancestors:", common_ancestors)
#todo Test on the array above for slice the dendrogram
# TODO: need to complete the tree consistency measurement

class DendrogramChildren(object):
    """Compute childen for a given dendogram node

    Parameters
    ----------
    ddata : dict
       data returned by scipy.cluster.hierarchy.dendrogram function
    """

    def __init__(self, ddata):
        self.icoord = np.array(ddata['icoord'])[:, 1:3]
        self.dcoord = np.array(ddata['dcoord'])[:, 1]
        self.icoord_min = self.icoord.min()
        self.icoord_max = self.icoord.max()
        self.leaves = np.array(ddata['leaves'])

    def query(self, node_id):
        """ Get all children for the node (specified by node_id) """
        mask = self.dcoord[node_id] >= self.dcoord

        def _interval_intersect(a0, a1, b0, b1):
            return a0 <= b1 and b0 <= a1

        # essentially intersection of lines from all children nodes
        sort_idx = np.argsort(self.dcoord[mask])[::-1]
        left, right = list(self.icoord[node_id])
        for ileft, iright in self.icoord[mask, :][sort_idx]:
            if _interval_intersect(ileft, iright, left, right):
                left = min(left, ileft)
                right = max(right, iright)

        extent = np.array([left, right])
        extent = (extent - self.icoord_min) * (len(self.leaves) - 1)
        extent /= (self.icoord_max - self.icoord_min)
        extent = extent.astype(int)
        extent = slice(extent[0], extent[1] + 1)
        return self.leaves[extent]
