import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score
from Utils import mkdir


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


"""
This is for a test case to construct a  directed graph object based
on ground truth hierarchy
"""


"""def hierarchy_tree(tree: nx.DiGraph(), target_folder = None):
    # Define the layout for the tree structure with top-down direction
    pos = nx.nx_agraph.graphviz_layout(tree, prog='dot', args="-Grankdir=TB")
    # Draw the tree structure
    file_path = "tree_dendrogram.pdf"
    target_path = os.path.join(target_folder, file_path)
    if os.path.exists(target_path) is False:
        plt.figure(figsize=(8, 6))
        nx.draw(tree, pos, with_labels=True, node_color='lightblue', node_size=1500, arrows=True)  # pos,
        plt.title("Tree Structure")
        plt.savefig(target_path)
        plt.show()

"""


def present_children(tree, parent):
    # Print the related nodes (children)
    children = list(tree.successors(parent))
    print("Children:", children)


# Perform hierarchical clustering


def plot_tree(linkage_matrix, folder = None, node_labels=None):
    dendrogra = sch.dendrogram(linkage_matrix, labels=node_labels)
    file_path = "dendrogram.pdf"
    target_path = os.path.join(folder, file_path)
    if os.path.exists(target_path) is False:
        # Set plot labels and title
        plt.xlabel('Distance')
        plt.ylabel('Nouns')
        plt.title('Noun Dendrogram')
        # Display the plot
        plt.savefig(target_path)
        plt.show()
    return dendrogra


def dendrogram_To_DirectedGraph(encodings, linkage_matrix, labels,Tfolder):
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
    #hierarchy_tree(tree, Tfolder)
    return tree


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


def sliced_clusters(linkage_m: sch.linkage, threshold: float, data):
    clusters = sch.cut_tree(linkage_m, height=threshold)
    # print(clusters)
    clusters1 = clusters.flatten()
    custom_clusters = {}
    for i, cluster_label in enumerate(clusters1):
        if cluster_label not in custom_clusters:
            custom_clusters[cluster_label] = []
        custom_clusters[cluster_label].append(i)

    # Convert custom_clusters to a list of lists
    custom_clusters_list = [nodes for nodes in custom_clusters.values()]

    clusters_1d = np.ravel(clusters)
    silhouette_avg = silhouette_score(data, clusters_1d)

    # print(f"Silhouette coefficient: {silhouette_avg}")
    return silhouette_avg, custom_clusters


def best_clusters(dendrogram: sch.dendrogram, linkage_m: sch.linkage, data, low=-1.0):
    # Get the y-coordinates from 'dcoord'
    y_coords = dendrogram['dcoord']
    # Find the highest and lowest y-values
    highest_y = np.max(y_coords)
    lowest_y = np.min(y_coords)
    print("  highest_y, lowest_y ", highest_y, lowest_y)
    silhouette = -1
    best_threshold = 0.0
    best_clusters = None

    numbers_with_boundaries =np.linspace(lowest_y, highest_y, 12)
    print(numbers_with_boundaries[1:-1])
    if low == -1:
        for threshold in numbers_with_boundaries[1:-1]:
            silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data)
            print(silhouette_avg, len(custom_clusters))
            if silhouette_avg > silhouette:
                best_clusters = custom_clusters
                silhouette = silhouette_avg
                best_threshold = threshold

            else:
                continue
    else:

        silhouette, best_clusters = sliced_clusters(linkage_m, low, data)
    print("best silhouette, ", silhouette, len(best_clusters.keys()))

    return best_threshold, best_clusters


def slice_tree(tree: nx.DiGraph(), custom_clusters, node_labels, is_Label=True):
    """
    The following is for the inferr the cluster in the tree
    """
    cluster_closest_parent = {}
    cluster_ancestors = {}
    for cluster_id, nodes_index in custom_clusters.items():
        nodes = [node_labels[i] for i in nodes_index]
        # print(nodes)

        # Find common ancestors of nodes A, B, and D
        common_ancestors = set(tree.nodes)
        for node in nodes:
            common_ancestors = common_ancestors.intersection(nx.ancestors(tree, node))
        # Find the closest parent node
        closest_parent = None
        min_depth = float('inf')
        for ancestor in common_ancestors:
            depth_dict = nx.shortest_path_length(tree, ancestor)
            depth = min(depth_dict[node] for node in nodes)
            if depth < min_depth:
                min_depth = depth
                closest_parent = ancestor
        # print("Common Ancestors:", common_ancestors)
        # print("Closest Parent Node:", closest_parent)
        if is_Label:
            # cluster_closest_parent[closest_parent] = nodes
            # cluster_ancestors[tuple(common_ancestors)] = nodes
            cluster_closest_parent[tuple(nodes)] = closest_parent, common_ancestors
        else:
            # cluster_ancestors[tuple(common_ancestors)] = nodes_index
            cluster_closest_parent[tuple(nodes_index)] = closest_parent, common_ancestors
    return cluster_closest_parent


# encodings = np.random.rand(25, 3)  # Assume each noun has a 5-dimensional encoding
encodings = np.array([[0.15264832, 0.67790326, 0.35618801],
                      [0.07687365, 0.21208948, 0.23942163],
                      [0.83904511, 0.91205178, 0.39408256],
                      [0.49731815, 0.79465782, 0.77706572],
                      [0.41300549, 0.69348555, 0.80835147],
                      [0.27307791, 0.81304529, 0.51075196],
                      [0.92478539, 0.33409272, 0.69586836],
                      [0.19984104, 0.52642224, 0.10401238],
                      [0.47016623, 0.4690633, 0.6132469],
                      [0.64690087, 0.98867187, 0.42791464],
                      [0.57888858, 0.08689729, 0.30285491],
                      [0.24681078, 0.64459448, 0.48688031],
                      [0.71598771, 0.96611438, 0.23196555],
                      [0.05359564, 0.41624766, 0.76461378],
                      [0.19783982, 0.21246817, 0.28792796],
                      [0.26966902, 0.27370917, 0.70342812],
                      [0.91653158, 0.96413556, 0.15612174],
                      [0.24759889, 0.74969501, 0.44362298],
                      [0.38557235, 0.76425578, 0.87203644],
                      [0.56123633, 0.01721618, 0.30774663],
                      [0.82745475, 0.84023844, 0.79468905],
                      [0.24703718, 0.66320813, 0.27354134],
                      [0.78698088, 0.78636944, 0.90183845],
                      [0.72581221, 0.35837165, 0.70863178],
                      [0.13436005, 0.85825595, 0.10523331]])
#print(encodings)
"""encodings = np.array([[0.29661852, 0.95912228, 0.25205271],
                      [0.98564082, 0.37565498, 0.73105763],
                      [0.93450012, 0.12862963, 0.86722713],
                      [0.14059451, 0.73131089, 0.59513037],
                      [0.52330052, 0.45879583, 0.04439388],
                      [0.7673861, 0.73452903, 0.14095513],
                      [0.15567008, 0.95688681, 0.96494895],
                      [0.79879999, 0.56013694, 0.9268367],
                      [0.98665682, 0.55432251, 0.75036973],
                      [0.92581121, 0.68291704, 0.15144076]])"""
# node_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
"""node_labels = [chr(ord('a') + i) for i in range(25)]
ground_label_name = "groundTruth.csv"
linkage_matrix = sch.linkage(encodings, method='complete', metric='euclidean')
folder = "fig/Example"
mkdir(folder)
result_folder = "result/Example"
mkdir(result_folder)
dendrogra = plot_tree(linkage_matrix, folder, node_labels=node_labels)
tree_test = dendrogram_To_DirectedGraph(encodings, linkage_matrix, node_labels,folder)
thre1, clusters = best_clusters(dendrogra, linkage_matrix, encodings)
parent_nodes = slice_tree(tree_test, clusters, node_labels)


file_path = "data.pkl"
os.path.join(result_folder,file_path )
try:
    # Open the file in binary write mode
    with open(file_path, 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump((dendrogra,linkage_matrix,tree_test,parent_nodes), file)
    print(f"Data successfully stored in '{file_path}'.")
except Exception as e:
    print(f"Error occurred while storing the pickle file: {e}")

thre2, clusters_sub = best_clusters(dendrogra, linkage_matrix, encodings, low=thre1 - 0.3)
sub_parent_nodes = slice_tree(tree_test, clusters_sub, node_labels)
"""






datasets = ["WDC","TabFact"] #, "TabFact"
for dataset in datasets:
    EMBEDDING_FOLDER = os.path.join(os.getcwd(),"result/embedding/starmie/vectors",dataset)
    for embedding in [i for i in os.listdir(EMBEDDING_FOLDER) if i.endswith("pkl")]:
        with open(os.path.join(EMBEDDING_FOLDER, embedding), 'rb') as file:
            data = pickle.load(file)
        node_labels=[i[0] for i in data]
        vectors=np.array([np.mean(i[1], axis=0) for i in data])

        #print(node_labels,vectors)
        ground_label_name = "groundTruth.csv"
        data_path = os.path.join(os.getcwd(),"datasets",dataset,ground_label_name )
        ground_truth_csv = pd.read_csv(data_path, encoding='latin1')

        labels = {ground_truth_csv.iloc[i,0]:ground_truth_csv.iloc[i,1] for i in range(0,len(ground_truth_csv))}

        #print(node_labels)
        linkage_matrix = sch.linkage(vectors, method='complete', metric='euclidean')
        folder = os.path.join("fig/Example",dataset,"clustering", embedding[:-4] )
        mkdir(folder)
        result_folder  = os.path.join("result/starmie/",dataset,"clustering" )
        mkdir(result_folder)
        dendrogra = plot_tree(linkage_matrix, folder, node_labels=node_labels)
        tree_test = dendrogram_To_DirectedGraph(vectors, linkage_matrix, node_labels, folder)
        thre1, clusters = best_clusters(dendrogra, linkage_matrix, vectors)
        parent_nodes = slice_tree(tree_test, clusters, node_labels)


        file_path =os.path.join(result_folder, embedding[:-4]+"_data.pkl")
        try:
            # Open the file in binary write mode
            with open(file_path, 'wb') as file:
                # Dump the data into the pickle file
                pickle.dump((dendrogra, linkage_matrix, tree_test, parent_nodes), file)
            print(f"Data successfully stored in '{file_path}'.")
        except Exception as e:
            print(f"Error occurred while storing the pickle file: {e}")

        #break
"""for clusters, parent_n in parent_nodes.items():
    for sub_clusters, parentNodes in sub_parent_nodes.items():
        if set(sub_clusters).issubset(set(clusters)):
            print("sub_clusters", clusters, parent_n, sub_clusters, parentNodes)
"""
# todo Test on the array above for slice the dendrogram
# TODO: need to complete the tree consistency measurement
