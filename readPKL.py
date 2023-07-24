import os.path
import pickle
import sys
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score
from Utils import mkdir
from collections import Counter


def most_frequent(list1):
    count = Counter(list1)
    most_common = count.most_common(2)

    # 获取前两个元素的出现次数
    if len(most_common) > 1:
        _, frequency = most_common[0]
        _, next_frequency = most_common[1]
        first_element_count = count[list1[0]]
        # 检查前两个元素的出现次数是否相同
        is_Same = True
        if frequency == next_frequency:
            # 提取出现频率最高的前两个元素
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
        # Display the plot
        plt.savefig(target_path)
        plt.show()
    return dendrogra


def dendrogram_To_DirectedGraph(encodings, linkage_matrix, labels):
    # Create a new NetworkX tree object
    tree = nx.DiGraph()
    # Iterate over the linkage matrix
    for i, link in enumerate(linkage_matrix):
        child_1, child_2, distance, _ = link
        parent_node = i + len(encodings)  # Assign unique IDs to parent nodes dist_matrix
        # Add edges to the tree object

        tree.add_edge(parent_node, int(child_1), length=distance)  # ,weight=1,capacity=15,
        tree.add_edge(parent_node, int(child_2), length=distance)
    # Assign labels to the tree nodes
    tree = nx.relabel_nodes(tree, {i: label for i, label in enumerate(labels)})
    # hierarchy_tree(tree, Tfolder)
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


def best_clusters(dendrogram: sch.dendrogram, linkage_m: sch.linkage, data, estimate_num_cluster=0):  # , low=-1.0, t2=0
    clusters = []
    # Get the y-coordinates from 'dcoord'
    y_coords = dendrogram['dcoord']
    # Find the highest and lowest y-values
    highest_y = np.max(y_coords)
    lowest_y = np.min(y_coords)
    # print("  highest_y, lowest_y ", highest_y, lowest_y)
    silhouette = -1
    best_threshold = 0.0
    best_clusters = None
    gap = sys.maxsize
    numbers_with_boundaries = np.linspace(lowest_y, highest_y, 12)
    # print(numbers_with_boundaries[1:-1])
    # if low == -1:
    for threshold in numbers_with_boundaries[1:-1]:
        silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data)
        # print(silhouette_avg, len(custom_clusters))
        if silhouette_avg > silhouette:
            best_clusters = custom_clusters
            silhouette = silhouette_avg
            best_threshold = threshold

        else:
            continue
    # print("best silhouette, ", silhouette, len(best_clusters.keys()))
    # return best_threshold, best_clusters
    clusters.append((best_threshold, best_clusters))
    # else:
    # if low < highest_y:
    #   silhouette, best_clusters = sliced_clusters(linkage_m, low, data)
    #  print("best silhouette, ", silhouette, len(best_clusters.keys()))
    # return best_threshold, best_clusters
    # else:
    if estimate_num_cluster != 0:
        for threshold in numbers_with_boundaries[1:-1]:
            silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data)
            # if len(custom_clusters)-low < gap and len(custom_clusters)-low > 0:
            if threshold > best_threshold:
                if abs(len(custom_clusters) - estimate_num_cluster) < gap \
                        and 0.6 * estimate_num_cluster < len(custom_clusters) < len(clusters[-1][1]):
                    clusters.append((threshold, custom_clusters))
    """
    if t2 != 0:
        for threshold in numbers_with_boundaries[1:-1]:
            silhouette_avg, custom_clusters = sliced_clusters(linkage_m, threshold, data)
            if len(custom_clusters) > t2:
                continue
            else:
                if threshold <= best_threshold:
                    thre_cluster[threshold] = custom_clusters
                    print(threshold, len(custom_clusters))
    """

    return clusters


def slice_tree(tree: nx.DiGraph(), custom_clusters, node_labels, is_Label=True):
    """
    The following is for the inferring the cluster in the tree
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


def simplify_graph(original_graph, cluster_results):
    # Create a new copy of the original graph to modify
    simplified_graph = original_graph.copy()
    leaf_nodes_by_cluster = {}
    # Step 1: Iterate through each cluster and add direct edges between leaf nodes and closest parent nodes
    for cluster, (closest_parent, mutual_parent_nodes) in cluster_results.items():
        # Add direct edges between leaf nodes and closest parent node
        for parent_node in mutual_parent_nodes:
            leaf_nodes_by_cluster.setdefault(parent_node, set()).add(cluster)
    # Step 2: Remove nodes and edges not part of the clusters or the path to the closest parent node
    nodes_to_remove = set(original_graph.nodes()) - set(leaf_nodes_by_cluster)
    simplified_graph.remove_nodes_from(nodes_to_remove)
    for cluster, (closest_parent, mutual_parent_nodes) in cluster_results.items():
        for leaf_node in cluster:
            simplified_graph.add_edge(closest_parent, leaf_node)
            simplified_graph.nodes[leaf_node]['type'] = 'data'
            # print(simplified_graph.nodes[closest_parent]['type'],simplified_graph.nodes[leaf_node]['type'])

    print(simplified_graph)
    return simplified_graph


def print_tree_p(digraph, root_nodes):
    def dfs_print(node, depth):
        print("  " * depth + str(node))
        for child in digraph.successors(node):
            dfs_print(child, depth + 1)

    for root_node in root_nodes:
        dfs_print(root_node, 0)


def simple_tree_with_cluster_label(threCluster_dict, orginal_tree, ground_truth, table_names, data=None):
    layer_info_dict = {}
    Parent_nodes_h = {}
    simple_tree = None
    for index, (thre, clusters) in enumerate(threCluster_dict):
        if index == 0:
            start_time = time.time()
            parent_nodes = slice_tree(orginal_tree, clusters, table_names)  #
            end_time = time.time()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the best silhouette tree")

            # print(parent_nodes)
            simple_tree = simplify_graph(orginal_tree, parent_nodes)
            # print(simple_tree)
        start_time = time.time()
        parent_nodes = slice_tree(orginal_tree, clusters, table_names)  #
        layer_info_dict[index] = parent_nodes
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the tree")
        layer_num = len(threCluster_dict) - 1
        start_time_label = time.time()
        for cluster, (closest_parent, mutual_parent_nodes) in parent_nodes.items():
            simple_tree.nodes[closest_parent]['data'] = list(cluster)
            index_ground_truth = 0
            if index == layer_num:

                labels_data = [ground_truth[i][1] if isinstance(ground_truth, dict) else ground_truth[data.index(i)] for
                               i in cluster]
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node parent layer'
                index_ground_truth = 1
                Parent_nodes_h[cluster] = closest_parent
            else:
                labels_data = [ground_truth[i][0] if isinstance(ground_truth, dict) else ground_truth[data.index(i)] for
                               i in cluster]
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node'
            simple_tree.nodes[closest_parent]['label'] = most_frequent(labels_data)
            wrong_labels = {}
            for i in cluster:
                label_per = ground_truth[i][index_ground_truth] if isinstance(ground_truth, dict) else ground_truth[
                    data.index(i)]
                if label_per not in simple_tree.nodes[closest_parent]['label']:
                    wrong_labels[i] = label_per
            simple_tree.nodes[closest_parent]['Wrong_labels'] = wrong_labels
            label_cluster = simple_tree.nodes[closest_parent]['label']
            simple_tree.nodes[closest_parent]['Purity'] = "{:.3f}".format(
                (1 - len(wrong_labels.keys()) / len(cluster))/len(label_cluster))
        end_time_label = time.time()
        elapsed_time = end_time_label - start_time_label
        print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the tree")

    return simple_tree, layer_info_dict, Parent_nodes_h


def print_tree_labels(tree, ground_truth, data=None):
    for index, (node, attributes) in enumerate(tree.nodes(data=True)):
        # if 'type' in attributes and attributes['type'] == 'data cluster node parent layer':#data cluster node
        if 'type' in attributes and attributes['type'] != 'data':  # data cluster node
            print(f"Node {node}:")
            print(f"  data: {attributes['data']}")
            print(f"  label: {attributes['label']}")
            print(f"  type: {attributes['type']}")
            print(f"  Data labels: {[ground_truth[la] for la in attributes['data']]}") if isinstance(ground_truth,
                                                                                                     dict) else \
                print(f"  Data labels: {[ground_truth[data.index(la)] for la in attributes['data']]}")
            if len(attributes['Wrong_labels']):
                print(f"wrong  label: {attributes['Wrong_labels']}")
            print(f"Purity is : {attributes['Purity']}")
            wrong_labels = {}
            for la in attributes['data']:
                per_label = ''
                if attributes['type'] == 'data cluster node parent layer':
                    per_label = ground_truth[la][1] if isinstance(ground_truth, dict) else \
                        ground_truth[data.index(la)]
                    """if per_label != attributes['label']:
                        wrong_labels[la] = per_label"""
                if attributes['type'] == 'data cluster node':
                    per_label = ground_truth[la][0] if isinstance(ground_truth, dict) else \
                        ground_truth[data.index(la)]
                    """if per_label != attributes['label']:
                        wrong_labels[la] = per_label"""

                if per_label not in attributes['label']:
                    wrong_labels[la] = per_label
            if len(wrong_labels) != 0:
                print(f" Wrong Data labels: {wrong_labels}", "Purity rate is ",
                      1 - len(wrong_labels) / len(attributes['data']))


def print_path_label(tree, layer_info, Parent_nodes, class_dict):
    all_pathCompatibility = []
    for cluster, (closest_parent, mutual_parent_nodes) in layer_info[0].items():
        for clusterP, closest_parentP in Parent_nodes.items():
            if closest_parentP in mutual_parent_nodes:
                path = sorted(list(mutual_parent_nodes))
                path = path[0:path.index(closest_parentP) + 1]
                print(path, closest_parentP)
                MatchedElementPath = len(path) - 1
                MatchedElementGT = 0
                for i in path:
                    label = tree.nodes[i].get('label', 0)
                    if label != 0:
                        print(f" Node: {i}", f" cluster label: {label}",
                              f" Purity: {tree.nodes[i].get('Purity', 0)}"
                              f" Wrong labels: {tree.nodes[i].get('Wrong_labels', 0)}"
                              )
                        superLabels = tree.nodes[closest_parentP].get('label', 0)
                        if superLabels != 0:
                            for label_per in label:
                                for superLabel in superLabels:
                                    if label_per in class_dict[superLabel]:
                                        MatchedElementGT += 1
                                        break  # Break out of the inner loop when the condition is met
                                else:
                                    continue
                                break

                    else:
                        MatchedElementPath -= 1
                if MatchedElementPath != 0:
                    pathCompatibility = MatchedElementGT / MatchedElementPath
                    all_pathCompatibility.append(pathCompatibility)

    print(all_pathCompatibility)
    tree_consistency_score = "{:.3f}".format(sum(all_pathCompatibility) / len(all_pathCompatibility))
    print("the metric is ", tree_consistency_score, "all_path_number is ", len(tree_consistency_score))
    return tree_consistency_score


def test_tree_consistency_metric(embedding_file, dataset):
    if dataset == "Example":
        ground_truth = ['A', 'C', 'C', 'B', 'E', 'E', 'C', 'E', 'D', 'A', 'D', 'E', 'C', 'A', 'A', 'D', 'C', 'D', 'D',
                        'B', 'B', 'A', 'B', 'E', 'B']

        # Print the resulting list of labels
        node_labels = [chr(ord('a') + i) for i in range(25)]

        ground_label_name = "groundTruth.csv"
        linkage_matrix = sch.linkage(embedding_file, method='complete', metric='euclidean')
        folder = "fig/Example"
        mkdir(folder)
        dendrogra = plot_tree(linkage_matrix, folder, node_labels=node_labels)
        tree_test = dendrogram_To_DirectedGraph(embedding_file, linkage_matrix, node_labels)
        threCluster_dict = best_clusters(dendrogra, linkage_matrix, embedding_file, estimate_num_cluster=3)
        simple_tree, layer_info_dict, Parent_nodes_h = simple_tree_with_cluster_label(threCluster_dict, tree_test,
                                                                                      ground_truth, node_labels,
                                                                                      data=node_labels)
        # 遍历并打印每个节点的 'data' 和 'label' 属性
        print_tree_labels(simple_tree, ground_truth, data=node_labels)
        print(layer_info_dict[0])
        for cluster, (closest_parent, mutual_parent_nodes) in layer_info_dict[0].items():
            for clusterP, closest_parentP in Parent_nodes_h.items():
                if closest_parentP in mutual_parent_nodes:
                    path = sorted(list(mutual_parent_nodes))
                    path = path[0:path.index(closest_parentP) + 1]
                    print(path)
                    for i in path:
                        label = simple_tree.nodes[i].get('label', 0)
                        print(i, label)
        print_path_label(simple_tree, layer_info_dict, Parent_nodes_h)
    else:

        ground_label_name1 = "groundTruth.csv"
        data_path = os.path.join(os.getcwd(), "datasets", dataset, ground_label_name1)
        ground_truth_csv = pd.read_csv(data_path, encoding='latin1')
        ground_truth = {ground_truth_csv.iloc[i, 0]: (ground_truth_csv.iloc[i, 4], ground_truth_csv.iloc[i, 5])
                        for i in range(0, len(ground_truth_csv))}

        class_dict = {key: group['class'].tolist() for key, group in ground_truth_csv.groupby('superclass')}
        # print(class_dict)
        up_layer_clusters_num = ground_truth_csv.iloc[:, 5].unique()
        table_with_available_labels = ground_truth_csv.iloc[:, 0].unique()
        EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", dataset)
        with open(os.path.join(EMBEDDING_FOLDER, embedding_file), 'rb') as file:
            data = pickle.load(file)
        data = [i for i in data if i[0] in table_with_available_labels]
        table_names = [i[0] for i in data if i[0] in ground_truth.keys()][0:5000]
        encodings = np.array([np.mean(i[1], axis=0) for i in data if i[0] in ground_truth.keys()][0:5000])

        linkage_matrix = sch.linkage(encodings, method='complete', metric='euclidean')
        folder = "fig/"+dataset
        mkdir(folder)
        result_folder = os.path.join("result/Valerie",dataset)
        mkdir(result_folder)
        dendrogra = plot_tree(linkage_matrix, folder, node_labels=table_names)
        tree_test = dendrogram_To_DirectedGraph(encodings, linkage_matrix, table_names)
        start_time = time.time()
        threCluster_dict = best_clusters(dendrogra, linkage_matrix, encodings,
                                         estimate_num_cluster=len(up_layer_clusters_num))
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")
        simple_tree, layer_info_dict, Parent_nodes_h = \
            simple_tree_with_cluster_label(threCluster_dict, tree_test, ground_truth, table_names)

        print_tree_labels(simple_tree, ground_truth)
        print_path_label(simple_tree, layer_info_dict, Parent_nodes_h, class_dict)

        file_path = os.path.join(result_folder, embedding_file[0:-4] + "_results.pkl")
        with open(file_path, 'wb') as file:
            # Dump the data into the pickle file
            pickle.dump((dendrogra, linkage_matrix, threCluster_dict, simple_tree), file)


def test(dataset):
    if dataset == "Example":

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
        test_tree_consistency_metric(encodings, dataset)

    else:
        EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", dataset)
        for embedding in [i for i in os.listdir(EMBEDDING_FOLDER) if i.endswith("pkl") and
                                                                     "roberta" in i]:
            test_tree_consistency_metric(embedding, dataset)
            break


test("TabFact")  # TabFact Example
# The below code is just for testing

""" 
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

# Create the initial list with labels repeated the required number of times
ground_truth0 = ['A', 'C', 'C', 'B', 'E', 'E', 'C', 'E', 'D', 'A', 'D', 'E', 'C', 'A', 'A', 'D', 'C', 'D', 'D', 'B',
                 'B', 'A', 'B', 'E', 'B']
# Print the resulting list of labels
node_labels = [chr(ord('a') + i) for i in range(25)]
ground_label_name = "groundTruth.csv"
linkage_matrix = sch.linkage(encodings, method='complete', metric='euclidean')
folder = "fig/Example"
mkdir(folder)
result_folder = "result/Example"
mkdir(result_folder)
dendrogra = plot_tree(linkage_matrix, folder, node_labels=node_labels)
tree_test = dendrogram_To_DirectedGraph(encodings, linkage_matrix, node_labels)

threCluster_dict0 = best_clusters(dendrogra, linkage_matrix, encodings, estimate_num_cluster=3)
simple_tree = None
layer_info1 = {}
Parent_nodes1 = {}
for index, (thre, clusters) in enumerate(threCluster_dict0):
    if index == 0:
        parent_nodes = slice_tree(tree_test, clusters, node_labels)  #
        simple_tree = simplify_graph(tree_test, parent_nodes)
        # print(simple_tree)

    parent_nodes = slice_tree(tree_test, clusters, node_labels)
    layer_info1[index] = parent_nodes
    for cluster, (closest_parent, mutual_parent_nodes) in parent_nodes.items():
        simple_tree.nodes[closest_parent]['data'] = list(cluster)
        labels_data = [ground_truth0[node_labels.index(i)] for i in cluster]
        simple_tree.nodes[closest_parent]['label'] = most_frequent(labels_data)
        if index == len(threCluster_dict0) - 1:
            simple_tree.nodes[closest_parent]['type'] = 'data cluster node parent layer'
        else:
            simple_tree.nodes[closest_parent]['type'] = 'data cluster node'
        if index == len(threCluster_dict0) - 1:
            Parent_nodes1[cluster] = closest_parent
    # 遍历并打印每个节点的 'data' 和 'label' 属性
for index, (node, attributes) in enumerate(simple_tree.nodes(data=True)):
  if 'type' in attributes and attributes['type'] !='data' : #'data cluster node parent layer'
    print(f"Node {node}:")
    if 'data' in attributes:
        print(f"  data: {attributes['data']}")
    if 'label' in attributes:
        print(f"  label: {attributes['label']}")
        print(f"  type: {attributes['type']}")
        print(f"  Data labels: {[ground_truth0[node_labels.index(la)] for la in attributes['data']]}")
        wrong_labels = []
        for la in attributes['data']:
            if ground_truth0[node_labels.index(la)] != attributes['label']:
                wrong_labels.append(ground_truth0[node_labels.index(la)])
        print(f" Wrong Data labels: {wrong_labels}")
print_path_label(simple_tree,layer_info1, Parent_nodes1)
print(layer_info1[0])
"""
"""root_nodes = [node for node, in_degree in simple_tree.in_degree() if in_degree == 0]
print(root_nodes)
# Print the tree structure#
print_tree(simple_tree, root_nodes)"""

# print_tree(tree_test, root_node2)
"""
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
"""
datasets = ["TabFact"]  # "WDC",





for dataset in datasets:
    EMBEDDING_FOLDER = os.path.join(os.getcwd(),"result/embedding/starmie/vectors",dataset)
    for embedding in [i for i in os.listdir(EMBEDDING_FOLDER) if i.endswith("pkl") and
                                                                 "sbert" in i]:
        with open(os.path.join(EMBEDDING_FOLDER, embedding), 'rb') as file:
            data = pickle.load(file)
        node_labels=[i[0] for i in data]
        vectors=np.array([np.mean(i[1], axis=0) for i in data])
        print(embedding)

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
for clusters, parent_n in parent_nodes.items():
    for sub_clusters, parentNodes in sub_parent_nodes.items():
        if set(sub_clusters).issubset(set(clusters)):
            print("sub_clusters", clusters, parent_n, sub_clusters, parentNodes)

# todo Test on the array above for slice the dendrogram
# TODO: need to complete the tree consistency measurement
for dataset in datasets:
    ground_label_name = "groundTruth.csv"
    data_path = os.path.join(os.getcwd(), "datasets", dataset, ground_label_name)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin1')
    clusters_num = ground_truth_csv['Label'].unique() if dataset == "WDC" else ground_truth_csv['label'].unique()
    print(len(clusters_num))
    labels = {ground_truth_csv.iloc[i, 0]: ground_truth_csv.iloc[i, 1] for i in range(0, len(ground_truth_csv))}
    FOLDER = os.path.join(os.getcwd(), "result/starmie/", dataset, "clustering")
    EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", dataset)
    for results in [i for i in os.listdir(FOLDER) if i.endswith("data.pkl")]:
        print(results)
        with open(os.path.join(EMBEDDING_FOLDER, results[:-9] + ".pkl"), 'rb') as file:
            data = pickle.load(file)
        node_labels = [i[0] for i in data]
        vectors = np.array([np.mean(i[1], axis=0) for i in data])
        with open(os.path.join(FOLDER, results), 'rb') as file:
            dendrogra, linkage_matrix, tree_test, parent_nodes = pickle.load(file)
        print("parent_nodes", len(parent_nodes.keys()))
        clusters = best_clusters(dendrogra, linkage_matrix, vectors, low=len(clusters_num),
                                        t2=len(parent_nodes.keys()))
        result_nodex = {}
        for threhold, clusters_sub in clusters.items():
            hnodes = slice_tree(tree_test, clusters_sub, node_labels)
            result_nodex[threhold] = (clusters_sub, hnodes)
        result_folder = os.path.join("result/starmie/", dataset, "clustering")
        file_path = os.path.join(result_folder, results[:-9][:-4] + "_data1.pkl")
        try:
                # Open the file in binary write mode
                with open(file_path, 'wb') as file:
                    # Dump the data into the pickle file
                    pickle.dump(result_nodex, file)
                print(f"Data successfully stored in '{file_path}'.")
        except Exception as e:
                print(f"Error occurred while storing the pickle file: {e}")


for dataset in datasets:
    ground_label_name = "groundTruth.csv"
    data_path = os.path.join(os.getcwd(), "datasets", dataset, ground_label_name)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin1')
    clusters_num = ground_truth_csv['Label'].unique() if dataset == "WDC" else ground_truth_csv['label'].unique()
    print(len(clusters_num))
    labels = {ground_truth_csv.iloc[i, 0]: ground_truth_csv.iloc[i, 1] for i in range(0, len(ground_truth_csv))}

    FOLDER = os.path.join(os.getcwd(), "result/starmie/", dataset, "clustering")
    EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", dataset)
    results_list1 = [i for i in os.listdir(FOLDER) if i.endswith("data.pkl")]
    results_list2 = [i for i in os.listdir(FOLDER) if i.endswith("data1.pkl")]
    for i in range(0, len(results_list1)):
        print(results_list1[i], results_list2[i])
        with open(os.path.join(EMBEDDING_FOLDER, results_list1[i][:-9] + ".pkl"), 'rb') as file:
            data = pickle.load(file)
        node_labels = [i[0] for i in data]
        vectors = np.array([np.mean(i[1], axis=0) for i in data])
        with open(os.path.join(FOLDER, results_list1[i]), 'rb') as file:
            dendrogra, linkage_matrix, tree_test, parent_nodes = pickle.load(file)
        with open(os.path.join(FOLDER, results_list2[i]), 'rb') as file:
            result_nodex = pickle.load(file)
        print("parent_nodes", len(result_nodex))
       # result_nodex_label = {}
        nodes_la = []
        for thre, cluster_thre in result_nodex.items():
            sub_nodes = cluster_thre
            if len(cluster_thre[0]) < 4000:
                #print(cluster_thre[0])
                if len(cluster_thre[0]) > 1000:
                    ground_label_name = "gt.csv"
                    data_path = os.path.join(os.getcwd(), "datasets", dataset, ground_label_name)
                    ground_truth_csv = pd.read_csv(data_path, encoding='latin1')

                    labels = {ground_truth_csv.iloc[i, 0]: (ground_truth_csv.iloc[i, 1], ground_truth_csv.iloc[i, 2])
                              for i in
                              range(0, len(ground_truth_csv))}


                cluster_labels = {}
                for cluster_id, tables in cluster_thre[0].items():
                    label_tables = [labels[node_labels[i]] for i in tables if node_labels[i] in labels.keys()]


                   


                    if len(label_tables) != 0:
                        label = most_frequent(label_tables)
                        try:
                            print(label)
                        except UnicodeEncodeError:
                            # If there is an encoding error, print to sys.stdout with a specific encoding
                            if isinstance(label,str):
                                    print(label.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
                            else:
                                print(label[0].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

                        cluster_labels[cluster_id] = tables,label
                if len(cluster_thre[0])>1000:
                    nodes_la.append((thre,cluster_labels))


"""
