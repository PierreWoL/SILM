from collections import deque
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


# 要查询的节点
def compute_max_distance(graph, start_node, target_node):
    visited = set()
    queue = deque([(start_node, 0)])  # (节点, 距离)
    max_distance = -1

    while queue:
        node, distance = queue.popleft()

        if node not in visited:
            visited.add(node)
            max_distance = max(max_distance, distance)

            if node == target_node:
                return max_distance

            for successor in graph.successors(node):
                queue.append((successor, distance + 1))

    return max_distance


def compute_maxD(ancestors, graph, target_node):
    max_distance = -1
    NodeMax = None
    dict_dis = {}
    for an in ancestors:
        shortest_path_lengths = nx.shortest_path_length(graph.reverse(), target=an, source=target_node)

        if shortest_path_lengths in dict_dis.keys():
            dict_dis[shortest_path_lengths].append(an)
        else:
            dict_dis[shortest_path_lengths] = [an]

    return dict_dis[max(list(dict_dis.keys()))]


# 找到父节点共同的最近子节点

# mutuals = close_child(tree, parent_top_per, type_low)
def close_child(G, nodes, child, is_mutual=True):
    # common_elements = set.intersection(*sets)
    dict_distance = {}
    node_d = {}
    child_set = [set(nx.descendants(G, node)) for node in nodes]
    if child_set != []:
        common_descendants = set.intersection(*child_set)
        for descendant in common_descendants:
            if is_mutual is True:
                distances = [nx.shortest_path_length(G, source=parent_node, target=descendant)
                             for parent_node in nodes if nx.has_path(G, descendant, child)]
            else:
                distances = [nx.shortest_path_length(G, source=parent_node, target=descendant)
                             for parent_node in nodes]
            if len(distances) > 0:
                avg_distance = sum(distances) / len(distances)
                node_d[descendant] = avg_distance
                if avg_distance in dict_distance.keys():
                    dict_distance[avg_distance].append(descendant)
                else:
                    dict_distance[avg_distance] = [descendant]

        min_dist = min(list(dict_distance.keys()))
        sorted_dict = dict(sorted(node_d.items(), key=lambda item: item[1]))
        sorted_distance = dict(sorted(dict_distance.items(), key=lambda item: item[0]))

        return min_dist, sorted_distance, node_d
    else:
        common_descendants = nodes
        return None


def close_parent(G, parents, children):
    node_d = {}
    dict_distance = {}
    for parent in parents:
        distances = [nx.shortest_path_length(G, source=parent, target=child) for child in children]
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
            node_d[parent] = avg_distance
            if avg_distance in dict_distance.keys():
                dict_distance[avg_distance].append(parent)
            else:
                dict_distance[avg_distance] = [parent]

    min_dist = min(list(dict_distance.keys()))
    sorted_dict = dict(sorted(node_d.items(), key=lambda item: item[1]))
    sorted_distance = dict(sorted(dict_distance.items(), key=lambda item: item[0]))
    return min_dist, sorted_distance, sorted_dict


def most_common_element_and_count(ancestors):
    # Combine all elements from all sets
    all_ancestors = [anc for ancestors_set in ancestors for anc in ancestors_set]
    common_ancestor = Counter(all_ancestors).most_common(1)[0][0]
    count_sets_with_most_common = sum(1 for s in ancestors if common_ancestor in s)
    return common_ancestor, count_sets_with_most_common


def lowest_common_ancestor_majority(graph, cluster_elements):
    ancestors = [set(nx.ancestors(graph, node)) | {node} for node in cluster_elements]
    common_ancestors = set.intersection(*ancestors)

    if not common_ancestors:
        # Finding the most common ancestor when there is no mutual ancestor
        common_ancestor, count_sets_with_most_common = most_common_element_and_count(ancestors)
    else:
        min_dist, sorted_distance, node_d = close_parent(graph, common_ancestors, cluster_elements)
        common_ancestor = list(node_d.keys())[0]
        count_sets_with_most_common = len(cluster_elements)
    return common_ancestor, count_sets_with_most_common


def count_descendants(graph, node):
    Sum = sum(graph.nodes[_]["weight"] for _ in nx.descendants(graph, node))
    return Sum+graph.nodes[node]["weight"]


def consistency_of_cluster(graph, cluster_elements):
    lca, cluster_length_valid = lowest_common_ancestor_majority(graph, cluster_elements)
    num_elements_in_lca_and_subtypes = count_descendants(graph, lca)
    return cluster_length_valid / num_elements_in_lca_and_subtypes


def at():
    # Creating a sample directed graph
    G = nx.DiGraph()
    edges = [('A', 'B'), ('B', 'D'), ('D', 'J'), ('D', 'K'), ('B', 'E'), ('E', 'L'), ('E', 'M'),
             ('A', 'C'), ('C', 'F'), ('F', 'O'), ('F', 'N'), ('C', 'G'), ('G', 'P'), ('G', 'Q'), ('H', 'I')]
    G.add_edges_from(edges)
    nx.set_node_attributes(G, 0, 'weight')
    # Example clusters

    cluster_1_elements = ['J', 'K', 'J', 'K', 'L', 'L', 'M', 'M', 'N', 'N']  # 'H', 'H',
    cluster_2_elements = ['O', 'O', 'N', 'Q', 'Q', 'H']

    def addWeight(cluster):
        for i in cluster:
            G.nodes[i]["weight"] += 1

    addWeight(cluster_1_elements)
    addWeight(cluster_2_elements)

    # Calculating consistency
    consistency_1 = consistency_of_cluster(G, cluster_1_elements)
    consistency_2 = consistency_of_cluster(G, cluster_2_elements)
    # print(f"Consistency of cluster 1: {consistency_1:.3f}")
    # print(f"Consistency of cluster 2: {consistency_2:.3f}")


"""

The following is for adding weight (ground truth table number) in the type hierarchy
"""

"""import pickle
import os

WDC_graph_path = "datasets/WDC/graphGroundTruth.pkl"
WDC_Gt_path = "datasets/WDC/groundTruth.csv"
F_cluster = open(os.path.join(WDC_graph_path), 'rb')
WDC_graph = pickle.load(F_cluster)
nx.set_node_attributes(WDC_graph, 0, 'weight')
ground_truth_df = pd.read_csv(WDC_Gt_path)

ground_truth_df = ground_truth_df.dropna()

dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"), ground_truth_df.iloc[:, 1]))
#print(dict_gt)
for i in dict_gt.values():
    WDC_graph.nodes[i]["weight"] = WDC_graph.nodes[i]["weight"]+1
    weight = WDC_graph.nodes[i]["weight"]
for i in set(list(dict_gt.values())):
    weight = WDC_graph.nodes[i]["weight"]
    print(f"node {i} now weight is {weight}")

sum_number = sum(WDC_graph.nodes[_]["weight"] for _ in set(list(dict_gt.values())))
print(sum_number)

with open(WDC_graph_path, "wb") as file:
    pickle.dump(WDC_graph, file)"""

"""print(WDC_graph)
graph_layout = nx.drawing.nx_agraph.graphviz_layout(WDC_graph, prog="dot", args="-Grankdir=TB")
plt.figure(figsize=(23, 23))
nx.draw(WDC_graph, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
plt.show()"""
