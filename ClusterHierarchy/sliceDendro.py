import os.path
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter


def most_frequent(list1):
    count = Counter(list1)
    most_common = count.most_common(2)
    if len(most_common) > 1:
        _, frequency = most_common[0]
        _, next_frequency = most_common[1]
        first_element_count = count[list1[0]]
        is_Same = True
        if frequency == next_frequency:
            for element in count.values():
                if element != first_element_count:
                    most_common_elements = [item[0] for item in most_common]
                    is_Same = False
                    return most_common_elements
            if is_Same:
                return list(count.keys())

    return [count.most_common(1)[0][0]]


def check_frequency(list1):
    # Create a Counter object
    count = Counter(list1)

    # Get the count of the first element in the list
    first_element_count = count[list1[0]]

    # Compare the count of the first element with the counts of other elements
    for element in count.values():
        if element != first_element_count:
            return False

    return True


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


def hierarchy_tree(tree: nx.DiGraph(), target_folder=None):  #
    # Define the layout for the tree structure with top-down direction
    # Draw the tree structure
    graph_layout = nx.drawing.nx_agraph.graphviz_layout(tree, prog="dot", args="-Grankdir=TB")
    plt.figure(figsize=(25, 25))
    nx.draw(tree, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
    plt.savefig(target_folder)
    return tree


def present_children(tree, parent):
    # Print the related nodes (children)
    children = list(tree.successors(parent))
    print("Children:", children)


# Perform hierarchical clustering


def plot_tree(linkage_matrix, folder=None, node_labels=None):
    if node_labels != None:
        dendrogra = sch.dendrogram(linkage_matrix, labels=node_labels)
    else:
        dendrogra = sch.dendrogram(linkage_matrix)
    file_path = "dendrogram.pdf"
    target_path = os.path.join(folder, file_path)
    if os.path.exists(target_path) is False:
        # Set plot labels and title
        plt.xlabel('Distance')
        plt.ylabel('Nouns')
        plt.title('Noun Dendrogram')
        plt.xticks(rotation=20)
        # Display the plot
        plt.savefig(target_path)
        plt.show()
    return dendrogra


def dendrogram_To_DirectedGraph(encodings, linkage_matrix, labels):
    # Create a new NetworkX tree object
    tree = nx.DiGraph()
    # print(len(encodings))
    # Iterate over the linkage matrix
    for i, link in enumerate(linkage_matrix):
        child_1, child_2, distance, _ = link
        parent_node = i + len(encodings)  # Assign unique IDs to parent nodes dist_matrix
        # Add edges to the tree object
        tree.add_edge(parent_node, int(child_1), length=distance)  # ,weight=1,capacity=15,
        tree.add_edge(parent_node, int(child_2), length=distance)
    # Assign labels to the tree nodes
    tree = nx.relabel_nodes(tree, {i: label for i, label in enumerate(labels)})
    # hierarchy_tree(tree)
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

def sliced_clusters( linkage_m: sch.linkage, threshold: float, data,
                    customMatrix='euclidean'):
    clusters = sch.cut_tree(linkage_m, height=threshold)
    # print(clusters)
    clusters1 = clusters.flatten()
    custom_clusters = {}
    for i, cluster_label in enumerate(clusters1):
        if cluster_label not in custom_clusters:
            custom_clusters[cluster_label] = []
        custom_clusters[cluster_label].append(i)
    # Convert custom_clusters to a list of lists

    clusters_1d = np.ravel(clusters)
    if customMatrix == 'euclidean':
        silhouette_avg = silhouette_score(data, clusters_1d)
    else:
        silhouette_avg = np.mean(silhouette_samples(data, clusters_1d, metric=customMatrix))
    return silhouette_avg, custom_clusters


def findBestSilhouetteDendro(dendrogram: sch.dendrogram,linkage_m: sch.linkage, data, customMatrix='euclidean'):
    silhouette = -1
    best_clustersR = None
    best_threshold = 0.0
    numbers_with_boundaries = np.linspace(best_threshold, np.max(dendrogram['dcoord']), 20)
    for threshold in numbers_with_boundaries[1:-1]:
        try:
            silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data, customMatrix)
            # print(silhouette_avg, len(custom_clusters))
            if silhouette_avg > silhouette:
                best_clustersR = custom_clusters
                silhouette = silhouette_avg
                best_threshold = threshold
            else:
                continue
        except ValueError as e:
            #print(e)
            continue
    return best_threshold, best_clustersR


def best_clusters(dendrogram: sch.dendrogram, linkage_m: sch.linkage, data,
                  customMatrix='euclidean', sliceInterval=10, delta=3):  # , low=-1.0, t2=0  estimate_num_cluster=0,
    clusters = []
    # Get the y-coordinates from 'dcoord'
    y_coords = dendrogram['dcoord']
    # Find the highest and lowest y-values
    highest_y = np.max(y_coords)

    silhouette = -1
    best_threshold = 0.0
    best_clustersR = None

    numbers_with_boundaries = np.linspace(best_threshold, highest_y, sliceInterval)
    for threshold in numbers_with_boundaries[1:-1]:
        try:
            silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data, customMatrix)
            # print(silhouette_avg, len(custom_clusters))
            if silhouette_avg > silhouette:
                best_clustersR = custom_clusters
                silhouette = silhouette_avg
                best_threshold = threshold
            else:
                continue
        except ValueError as e:

            continue
    if silhouette != -1:
        #print("best silhouette, ", silhouette, len(best_clustersR.keys()))

        clusters.append((best_threshold, best_clustersR))
        range_lower = silhouette - delta if silhouette - delta > -1 else -1
        range_upper = silhouette + delta if silhouette + delta < 1 else 1

        numbers_with_boundaries = np.linspace(best_threshold, highest_y, sliceInterval)

        # if estimate_num_cluster != 0:
        for threshold in numbers_with_boundaries[1:-1]:
            try:
                silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data, customMatrix)

                if range_lower < silhouette_avg < range_upper:

                    if len(custom_clusters) < len(clusters[-1][1]):
                        clusters.append((threshold, custom_clusters))
            except:
                continue
    # print(f'the total layer number is {len(clusters)}')
    return clusters


def slice_tree(tree: nx.DiGraph(), custom_clusters, node_labels, is_Label=True):
    """
    The following is for the inferring the cluster in the tree
    """
    cluster_closest_parent = {}
    cluster_ancestors = {}
    for cluster_id, nodes_index in custom_clusters.items():
        nodes = [node_labels[i] for i in nodes_index]
        # Find common ancestors of nodes A, B, and D
        common_ancestors = set(tree.nodes)
        for node in nodes:
            common_ancestors = common_ancestors.intersection(nx.ancestors(tree, node))

        # Find the closest parent node
        closest_parent = None
        min_depth = float('inf')
        if len(common_ancestors) == 0:
            # print(nodes)
            closest_parent = nodes[0]
            common_ancestors = nodes
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
            if closest_parent is None:
                print(f"this has exceptions {nodes, common_ancestors}")
                for i in nodes:
                    print(nx.ancestors(tree, i))

        else:
            # cluster_ancestors[tuple(common_ancestors)] = nodes_index
            cluster_closest_parent[tuple(nodes_index)] = closest_parent, common_ancestors
    return cluster_closest_parent


def simplify_graph(original_graph, cur_cluster_results, lowerNode=None):
    # Create a new copy of the original graph to modify
    simplified_graph = original_graph.copy()
    leaf_nodes_by_cluster = {}
    delete_nodes = []
    if lowerNode is None:
        # Step 1: Iterate through each cluster and add direct edges between leaf nodes and closest parent nodes
        for cluster, (closest_parent, mutual_parent_nodes) in cur_cluster_results.items():
            # Add direct edges between leaf nodes and closest parent node
            for parent_node in mutual_parent_nodes:
                leaf_nodes_by_cluster.setdefault(parent_node, set()).add(cluster)
        # Step 2: Remove nodes and edges not part of the clusters or the path to the closest parent node
        nodes_to_remove = set(original_graph.nodes()) - set(leaf_nodes_by_cluster)
        simplified_graph.remove_nodes_from(nodes_to_remove)
        for cluster, (closest_parent, mutual_parent_nodes) in cur_cluster_results.items():
            for leaf_node in cluster:
                simplified_graph.add_edge(closest_parent, leaf_node)
                simplified_graph.nodes[leaf_node]['type'] = 'data'
        return simplified_graph
    else:
        for cluster, (closest_parent, mutual_parent_nodes) in cur_cluster_results.items():
            children_close_parent = nx.descendants(simplified_graph, closest_parent)
            children_close_parent = [i for i in children_close_parent if simplified_graph.out_degree(i) != 0]
            all_children = set()
            node_keep = [i for i in lowerNode if i in children_close_parent]
            for i in node_keep:
                all_children.update(nx.descendants(simplified_graph, i))
            intermediate_nodes = [i for i in children_close_parent if i not in node_keep and i not in all_children]
            if len(intermediate_nodes) > 0:
                delete_nodes.extend(intermediate_nodes)
                for node in node_keep:
                    simplified_graph.add_edge(closest_parent, node)
                    simplified_graph.remove_nodes_from(intermediate_nodes)
                    print(node, intermediate_nodes)
        has_intersection = any(item in delete_nodes for item in simplified_graph.nodes())
        if has_intersection is True:
            print(delete_nodes, list(set(delete_nodes).intersection(simplified_graph.nodes())))
        return simplified_graph


def print_tree_p(digraph, root_nodes):
    def dfs_print(node, depth):
        print("  " * depth + str(node))
        for child in digraph.successors(node):
            dfs_print(child, depth + 1)

    for root_node in root_nodes:
        dfs_print(root_node, 0)


def print_node(tree: nx.DiGraph(), node, printer=False):
    node_dict = {'Node': node, 'Type': tree.nodes[node].get('type', 0),
                 'cluster label': tree.nodes[node].get('label', 0),
                 'Purity': tree.nodes[node].get('Purity', 0),
                 'Data': tree.nodes[node].get('data', 0),
                 'Wrong Labels': tree.nodes[node].get('Wrong_labels', 0)}
    if printer is True:
        print(f" Node: {node}", f" Type: {tree.nodes[node].get('type', 0)} \n",
              f" cluster label: {tree.nodes[node].get('label', 0)} \n",
              f" Purity: {tree.nodes[node].get('Purity', 0)} \n"
              f" Data: {tree.nodes[node].get('data', 0)}")
        if len(tree.nodes[node].get('Wrong_labels', 0)):
            print(f" Wrong Labels: {tree.nodes[node].get('Wrong_labels', 0)}")
    return node_dict


def print_clusters_top_down(tree, layer_info, store_path=None):
    list_tree_info = []
    total_layer = len(layer_info.keys()) - 1
    # print(f"Total layer is : {total_layer + 1}")
    index_layer = total_layer
    while index_layer >= 0:
        # print(f"layer : {len(layer_info.keys()) - index_layer}",
        # f"nodes number: {len(layer_info[index_layer].items())}")  # index_layer + 1
        index_layer -= 1
    # print("Layer information ...")
    for cluster, (closest_parent, mutual_parent_nodes) in layer_info[total_layer].items():
        list_tree_info.append(print_node(tree, closest_parent))
        index = total_layer - 1
        # print("its sub-types are: ")
        while index >= 0:
            # print(f"Layer is : {index}")
            for cluster0, (closest_parent0, mutual_parent_nodes0) in layer_info[index].items():
                if closest_parent in mutual_parent_nodes0:
                    list_tree_info.append(print_node(tree, closest_parent0))
            index -= 1
    if store_path is not None:
        list_tree = pd.DataFrame(list_tree_info)
        list_tree.to_csv(store_path, index=False)


def purity_per_layer(tree, layer_info):
    purity_layers = {}
    total_layer = len(layer_info.keys()) - 1
    while total_layer >= 0:
        purity_layers[len(layer_info.keys()) - total_layer] = []
        for cluster, (closest_parent, mutual_parent_nodes) in layer_info[total_layer].items():
            purity_layers[len(layer_info.keys()) - total_layer].append(tree.nodes[closest_parent].get('Purity', 0))
        total_layer -= 1
    return purity_layers


def print_path_label(tree, layer_info, Parent_nodes):
    all_pathCompatibility = []
    ind, ind_o = 0, 0
    for cluster, (closest_parent, mutual_parent_nodes) in layer_info[0].items():
        path = sorted(list(mutual_parent_nodes))
        ind_o += 1
        for indexp, (clusterP, closest_parentP) in enumerate(Parent_nodes.items()):
            if closest_parentP in path:
                path = path[0:path.index(closest_parentP)]
                MatchedElementPath = len(path)
                MatchedElementGT = 0
                if len(path) > 0:
                    ind += 1
                    for i in path:
                        label = tree.nodes[i].get('label', 0)
                        if label != 0:
                            print(f" Node: {i}", f" Type: {tree.nodes[i].get('type', 0)} \n",
                                  f" cluster label: {label} \n",
                                  f" Purity: {tree.nodes[i].get('Purity', 0)} \n"
                                  f" Data: {tree.nodes[i].get('data', 0)} \n",
                                  f" Wrong Labels: {tree.nodes[i].get('data', 0)} \n",
                                  )
                            superLabels = tree.nodes[closest_parentP].get('label', 0)

                    print(ind_o, ind, indexp, path)
                    if MatchedElementPath != 0:
                        pathCompatibility = MatchedElementGT / MatchedElementPath
                        all_pathCompatibility.append(pathCompatibility)

