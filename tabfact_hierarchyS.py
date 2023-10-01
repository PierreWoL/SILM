# Iterating through groups and displaying rows
import ast
import os
import pickle

import numpy as np
import pandas as pd
from collections import deque

from matplotlib import pyplot as plt

from ClusterHierarchy.ClusterDecompose import find_frequent_labels

import networkx as nx

from clustering import data_classes

target_path = os.path.join(os.getcwd(), "datasets/TabFact/")


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
        # print(f"节点 {an}: {shortest_path_lengths}")
        if shortest_path_lengths in dict_dis.keys():
            dict_dis[shortest_path_lengths].append(an)
        else:
            dict_dis[shortest_path_lengths] = [an]

    return dict_dis[max(list(dict_dis.keys()))]


"""
node = 'national association football team'
ancestors = list(nx.ancestors(G, node))


print(f"{node} ancestor at the highest level {compute_maxD(ancestors,G,node)}")
ma = [i for i in ancestors if G.in_degree(i) == 0]
print(f"{node}  ancestor while degree is 0 {ma}")"""
# nodes_with_indegree_zero = [node for node in G.nodes if G.in_degree(node) == 0]
# print(len(nodes_with_indegree_zero))
# print(f'parent nodes are :')
"""for i in nodes_with_indegree_zero:
    print(f'{i}')"""
farthest_ancestor = None
max_distance = -1


def ground_truth_labels(tree):
    ground_label_name1 = "01SourceTables.csv"
    data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
    files = ground_truth_csv["fileName"]
    dict_gt = []
    for filename in files:
        row = ground_truth_csv[ground_truth_csv['fileName'] == filename]
        if row['class'].item() != ' ':
            classX = row["class"].iloc[0]
            # table_label = ast.literal_eval(classX)
            labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
            if filename in labels:
                table_label = pd.read_csv(
                    os.path.join(os.getcwd(), "datasets/TabFact/Label", filename))[
                    "classLabel"].unique()
                for loa in table_label:
                    ancestors = list(nx.ancestors(tree, loa))
                    top_level_typ = [i for i in ancestors if tree.in_degree(i) == 0]
                    row = {'fileName': filename, 'Lowest level type': str(loa),
                           'Top level type': str(top_level_typ)}
                    dict_gt.append(row)
                    print(f"fileName: {filename} node: {loa}, top level type {top_level_typ}")


            else:
                ancestors = list(nx.ancestors(tree, classX))
                top_level_type = [i for i in ancestors if tree.in_degree(i) == 0]
                if len(top_level_type) == 0:
                    ancestors = list(nx.ancestors(tree, row["superclass"].iloc[0]))
                    top_level_type = [i for i in ancestors if tree.in_degree(i) == 0]
                    if len(top_level_type) == 0:
                        top_level_type = row["superclass"].tolist()
                row = {'fileName': filename, 'Lowest level type': str([classX]),
                       'Top level type': str(top_level_type)}
                dict_gt.append(row)
                print(f"fileName: {filename} node: {[classX]}, top level type {top_level_type}")


abstract = ['object', 'result', 'temporal entity', 'inconsistency', 'noun', 'noun phrase', 'remains', 'use',
            'independent continuant', 'observable entity', 'artificial entity', 'natural physical object',
            'occurrence', 'relation', "group of physical objects", 'economic entity',
            'concrete object', 'three-dimensional object', 'part', 'geographic entity', 'artificial geographic entity',
            'source', 'group or class of physical objects', 'role', 'phenomenon', 'physical entity', 'means',
            'spatio-temporal entity', 'spatial entity', 'one-dimensional space', 'physical object',
            'continuant', 'collective entity', 'space object', 'type', 'information', 'anatomical entity',
            'output', 'abstract object', 'non-physical entity', 'integral', 'quantity', 'former entity',
            'occurrent', 'cause', 'idiom', 'lect', 'modification', 'alteration', 'control', 'consensus',
            'social relation',
            'social phenomenon', 'manifestation', 'work', 'source of information', 'knowledge type', 'action',
            'time interval', 'effect', 'class',
            'status', 'group of living things', 'agent', 'sign', 'content', 'converter', 'resource', 'metaclass',
            'unit',
            'interface', 'contributing factor', 'undesirable characteristic', 'structure', 'method', 'matter', 'change',
            'physical phenomenon', 'binary relation', 'building work', 'power', 'management', 'long, thin object',

            'definite integral', 'physical property', 'multi-organism process', 'data', 'multiset', 'line',
            'proper noun',
            'interaction', 'information resource', 'list', 'plan', 'scale', 'memory', 'social structure',
            'source text', 'open content', 'written work', 'strategy', 'group of humans', 'system', 'deformation',
            'representation', 'multicellular organismal process', 'operator', 'social system']  # 'series'

"""
for i in abstract:
    G.remove_node(i)
top_level_typ = [i for i in G.nodes if G.in_degree(i) == 0]
print(len(top_level_typ), "\n", top_level_typ)

"""


# 找到父节点共同的最近子节点


def close_child(G, nodes, child, is_mutual=True):
    # common_elements = set.intersection(*sets)

    closest_child = None
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
        # print(sorted_dict)
        return min_dist, dict_distance, node_d
    else:
        common_descendants = nodes
        return None


# print(lower.values())
# with open(os.path.join(target_path, "graphGroundTruth2.pkl"), "wb") as file:
#    pickle.dump(G_tree, file)

# TODO the following is for the generating the top -level clas in the dataframe

"""
ground_label_name1 = "column_gt.csv"
data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
gt_csv = pd.read_csv(data_path, encoding='latin-1')
for index in range(len(gt_csv)):
    fileName = gt_csv.iloc[index,1]
    matching_indices = ground_truth_csv.index[ground_truth_csv["fileName"] == fileName].tolist()
    row = ground_truth_csv.iloc[matching_indices,]
    if len(row)>0:

        gt_csv.iloc[index, 6] = row["Lowest Type"]
        gt_csv.iloc[index, 7] = row["Top-level type"]
    else:
        gt_csv.iloc[index, 6] = ""
        gt_csv.iloc[index, 7] = ""
gt_csv.to_csv(data_path, index=False)"""

# TODO the following is reducing the graph


new_dict = {}

with open(os.path.join(target_path, "graphGroundTruth2.pkl"), "rb") as file:
    G_tree = pickle.load(file)
removal = ['PhysicalActivity', 'Season', 'nonfiction', 'Manuscript', 'conceptual system', 'historical source', 'text',
           'Psychiatric', 'nonfiction', 'VisualArtwork']
#
lower = {i.lower(): i for i in G_tree.nodes()}

abstract = ['relation', 'aspect', 'type', 'object', 'result', 'spatio-temporal entity', 'role', 'relation',
            'means', 'role', 'time interval', 'effect', 'class', 'text', 'occurrent', 'narrative', 'behavior',
            'collective entity', 'information', 'part', 'temporal entity', 'inconsistency', 'status', 'system',
            'noun', 'noun phrase', 'remains', 'use', 'work', 'output', 'part of a work', 'periodic process', 'politics',
            'spacetime volume', 'facility', 'network', 'record', 'artificial entity', 'spatial entity', 'activity',
            'economic entity', 'sign', 'structure', 'structure', 'three-dimensional object', 'pattern',
            'individual', 'source', 'metaclass', 'lect', 'power', 'interaction', 'line', 'process', 'dyad',
            'control', 'single', 'unit', 'integral', 'effect', 'work', 'matter', 'change']
G_tree.remove_nodes_from(abstract)


def reduce_G(tree: nx.DiGraph()):
    labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
    ground_label_name1 = "01SourceTables.csv"
    data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')

    exists_lowest = []
    top_nodes_keep = set()
    for index, row in ground_truth_csv.iterrows():

        if row["fileName"] in labels:
            label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
            all_classes = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8')
            lowest_type = all_classes.iloc[:, 3]
            others = set(all_classes.iloc[:, 4:9].stack().unique())
            lowest_type_unqi = lowest_type.unique()
            for type_low in lowest_type_unqi:
                if type_low in tree.nodes:
                    if type_low not in exists_lowest:
                        exists_lowest.append(type_low)
                        parent_top_per = [item for item in nx.ancestors(tree, type_low) if
                                          tree.in_degree(item) == 0]
                        parent = [item for item in nx.ancestors(tree, type_low)]
                        intersect = others.intersection(set(parent))
                        mutuals = close_child(tree, parent_top_per, type_low)
                        #  print(  f" the top level {parent_top_per},\n the upper type {intersect}  ")# \n the parent {parent} ,, \n\n
                        if mutuals != None:
                            top_nodes_keep.update(set(mutuals[1][mutuals[0]]))
                            # print(f"lowest type {type_low}")
                            # print(  f" \n mutual closest is {mutuals[1][mutuals[0]]} \n\n")
                            continue
                        else:
                            # print(f"lowest type {type_low}")
                            if len(parent_top_per) != 0:
                                # print(f" \n NO MUTUAL {parent_top_per} \n\n")
                                top_nodes_keep.update({set(parent_top_per)})
                            else:
                                # print(f" \n NO MUTUAL {type_low} \n\n")
                                top_nodes_keep.update({type_low})
        else:
            if row["class"] != " ":
                superclass = row["class"]
                classX = row["superclass"]
                top_nodes_keep.update({superclass, classX})

    print("Nodes in the graph:", len(tree.nodes()))
    print(len(top_nodes_keep), top_nodes_keep)
    all_descendants = set()
    all_ancestors = set()
    for node in top_nodes_keep:
        if node in tree.nodes():
            descendants = nx.descendants(tree, node)
            ancestors = nx.ancestors(tree, node)
            all_descendants.update(descendants)
            all_ancestors.update(ancestors)

    # Filter ancestors based on your condition
    nodes_to_remove = {node for node in all_ancestors if node not in all_descendants and node in top_nodes_keep}

    # Remove filtered nodes from the graph
    tree.remove_nodes_from(nodes_to_remove)

    # Show the resulting graph
    print("Nodes in the graph:", len(tree.nodes()))
    print("top_node ", [i for i in tree.nodes() if tree.in_degree(i) == 0])
    return len(tree.nodes())


def converging_reduce(num, some_threshold):
    previous_value = 0

    for i in range(num):
        if i == 0:
            previous_value = reduce_G(G_tree)
            continue
        else:
            current_value = reduce_G(G_tree)
            rate_of_change = abs((current_value - previous_value) / current_value)
            previous_value = current_value
            if rate_of_change < some_threshold:
                print("Converging")
                with open(os.path.join(target_path, "graphGroundTruth2_reduced.pkl"), "wb") as file:
                    pickle.dump(G_tree, file)
                exceptions = []
                labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
                ground_label_name1 = "01SourceTables.csv"
                data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
                ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
                for index, row in ground_truth_csv.iterrows():
                    parent_top_pers = []
                    if row["fileName"] in labels:
                        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
                        all_classes = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8')
                        lowest_type = all_classes.iloc[:, 3]
                        others = set(all_classes.iloc[:, 4:9].stack().unique())
                        lowest_type_unqi = lowest_type.unique()
                        for type_low in lowest_type_unqi:
                            if type_low in G_tree.nodes:
                                parent_top_per = [item for item in nx.ancestors(G_tree, type_low) if
                                                  G_tree.in_degree(item) == 0]
                                parent_top_pers.extend(parent_top_per)
                                print(f" {type_low} the top level {parent_top_per}")
                        ground_truth_csv.iloc[index, 4] = lowest_type_unqi
                        ground_truth_csv.iloc[index, 5] = list(set(parent_top_pers))
                    else:
                        if row["class"] != " ":
                            type_low = row["class"]
                            if type_low in G_tree.nodes():
                                parent_top_per = [item for item in nx.ancestors(G_tree, type_low) if
                                                  G_tree.in_degree(item) == 0]
                                parent_top_pers.extend(parent_top_per)
                                print(f"{type_low}  the top level {parent_top_per} ")
                            ground_truth_csv.iloc[index, 4] = type_low
                            ground_truth_csv.iloc[index, 5] = list(set(parent_top_pers))
                ground_truth_csv.to_csv(os.path.join(target_path, "new_test_reduced1.csv"))

                break





def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)


def reduce_graph(tree: nx.DiGraph(), node_list):
    node_list_no_parent = [i for i in node_list if tree.in_degree(i) != 0]
    print(node_list_no_parent)
    descendants_list = []
    ancestors_list = []
    deletes = []
    bottom_nodes = [node for node in tree.nodes() if tree.out_degree(node) == 0]
    for bottom_node in bottom_nodes:
        ancster = [node for node in nx.ancestors(tree, bottom_node) if tree.in_degree(node) == 0]
        for an in ancster:
            paths_to_top = nx.all_simple_paths(tree, source=an, target=bottom_node)
            for path in paths_to_top:
                path_contains_list_node = any(node in path for node in node_list_no_parent)

                if path_contains_list_node:
                    for node in path:
                        descendants = [i for i in nx.descendants(tree.subgraph(path), node) if i not in node_list]
                        ancs = [i for i in nx.ancestors(tree.subgraph(path), node) if i not in node_list]
                        print(bottom_node, node, descendants, ancs)
                        descendants_list.extend(descendants)
                        ancestors_list.extend(ancs)
                else:
                    copy = [i for i in path if tree.in_degree(i) != 0]
                    # deletes.extend(copy)

    dupli_descendant = find_duplicates(descendants_list)
    dupli_ancestor = find_duplicates(ancestors_list)

    union_set = list(set(dupli_descendant) | set(dupli_ancestor))

    union_set.extend(node_list)
    deletes = [i for i in tree.nodes() if i not in union_set]
    for node_delete in deletes:
        tree.remove_node(node_delete)
    return tree
#converging_reduce(200, 0.05)

with open(os.path.join(target_path, "graphGroundTruth2_reduced.pkl"), "rb") as file:
    G_tree = pickle.load(file)
labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
ground_label_name1 = "01SourceTables.csv"
data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
top_class = {i: 0 for i in G_tree.nodes() if G_tree.in_degree(i) == 0}
print(len(top_class))
array_class = np.zeros((len(top_class), len(top_class)))
overlaps = []

for index, row in ground_truth_csv.iterrows():
    parent_top_pers = []
    lowest = None
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        all_classes = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8')
        lowest_type = all_classes.iloc[:, 3]
        others = set(all_classes.iloc[:, 4:9].stack().unique())
        lowest_type_unqi = lowest_type.unique()
        for type_low in lowest_type_unqi:
            if type_low in G_tree.nodes:
                parent_top_per = [item for item in nx.ancestors(G_tree, type_low) if
                                  G_tree.in_degree(item) == 0]
                parent_top_pers.extend(parent_top_per)
        lowest = lowest_type_unqi


    else:
        if row["class"] != " ":
            type_low = row["class"]
            if type_low in G_tree.nodes():
                parent_top_per = [item for item in nx.ancestors(G_tree, type_low) if
                                  G_tree.in_degree(item) == 0]
                parent_top_pers.extend(parent_top_per)
            lowest = type_low

    ground_truth_csv.iloc[index, 4] = lowest
    ground_truth_csv.iloc[index, 5] = list(set(parent_top_pers))


class_count = ground_truth_csv['superclass'].explode().value_counts()
print(class_count)
class_count = class_count[(class_count >= 10) & (class_count.index != ' ')]

# Plot
"""plt.figure(figsize=(25, 15))
plt.xlabel('Top Level Class')
plt.ylabel('Frequency')
plt.title('Distribution of Classes')
plt.xticks(fontsize=11)
class_count.plot(kind='bar')
plt.show()
"""
from upsetplot import UpSet
from collections import Counter
ground_truth_csv['superclass'] = ground_truth_csv['superclass'].apply(frozenset)
ground_truth_csv = ground_truth_csv[ground_truth_csv['superclass'] != frozenset()]
# Count occurrences of each unique set of classes
count_series = ground_truth_csv['superclass'].value_counts()
count_series = count_series[count_series >= 10]

# Extract unique sets and their counts
unique_sets = count_series.index.tolist()
unique_counts = count_series.values.tolist()

# If you want to convert the frozensets back to lists
unique_sets_as_lists = [list(s) for s in unique_sets]

print("Unique sets:", unique_sets_as_lists)
print("Counts:", unique_counts)


from upsetplot import from_memberships

example = from_memberships(
    unique_sets_as_lists,
     data=unique_counts
 )
from upsetplot import plot
plot(example)
from matplotlib import pyplot
pyplot.show()

# 创建一个新的 DataFrame
df_count = pd.DataFrame({
    'Unique_Sets': [str(s) for s in unique_sets_as_lists],
    'Counts': unique_counts
})

# 根据 Counts 列对 DataFrame 进行排序
df_count = df_count.sort_values('Counts', ascending=False)

# 画出条形图
plt.barh(df_count['Unique_Sets'], df_count['Counts'])
plt.figure(figsize=(25, 15))
plt.xticks(fontsize=11)
plt.xlabel('Counts')
plt.ylabel('Unique Sets')
plt.title('Count of Each Unique Set')
plt.show()