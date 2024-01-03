import ast
import pickle
import time
from argparse import Namespace
import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import readPKL as PKL
import scipy.cluster.hierarchy as sch
from ClusterHierarchy.JaccardMetric import JaccardMatrix
from Utils import mkdir
from clustering import data_classes
from interactiveFigure import draw_interactive_graph


def tables_of_node(tree: nx.DiGraph(), node):
    successors_of_node = list(nx.descendants(tree, node))
    tables = tree.nodes[node]['Tables']
    if successors_of_node:
        for successor in successors_of_node:
            tables.extend(tree.nodes[successor]['Tables'])
    return tables


def find_frequent_labels(ancestors: list, G: nx.DiGraph()):
    # All parent nodes
    indegrees = [G.in_degree(parent) for parent in ancestors]
    parents = {}
    for index, ele in enumerate(indegrees):
        if ele not in parents.keys():
            parents[ele] = [ancestors[index]]
        else:
            parents[ele].append(ancestors[index])
    # the most closest parent node
    min_element = min(indegrees)
    min_indices = [i for i, x in enumerate(indegrees) if x == min_element]
    topmost_parent = [ancestors[i] for i in min_indices]
    return topmost_parent, parents


def labels_most_fre(datas: dict):
    if len(datas) == 1:
        return list(datas.values())[0], 1
    # frequency of each label
    label_counter = Counter(label for labels_list in datas.values() for label in labels_list)
    # Mayjor voting label
    most_common_labels = label_counter.most_common()
    # most frequently appeared label
    max_frequency = most_common_labels[0][1]
    most_common_labels = [label for label, frequency in most_common_labels if frequency == max_frequency]
    return most_common_labels, max_frequency


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0
    jaccard_sim = len(intersection) / len(union)
    return jaccard_sim


def label_dict(tables: list, dataset="WDC"):
    target_path = f"datasets/{dataset}"
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)
    matched_node = []
    jaccardSim = 0
    for node, attrs in G.nodes(data=True):
        tables_node = tables_of_node(G, node)
        sim = jaccard_similarity(tables, tables_node)
        if sim >= jaccardSim:
            jaccardSim = sim
            matched_node = [node]
        elif sim == jaccardSim:
            matched_node.append(node)
    return matched_node, jaccardSim


def no_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if len(set1.intersection(set2)) == 0:
        return True
    else:
        return False


def updateNodeInfo(tree, clusterNode, tables, dataset):
    labels, sim = label_dict(list(tables), dataset=dataset)  # True

    tree.nodes[clusterNode]['label'] = labels
    tree.nodes[clusterNode]['tables'] = tables

    wrong_labels = {}
    # tree.nodes[clusterNode]['Wrong_labels'] = wrong_labels
    tree.nodes[clusterNode]['Purity'] = sim


def simple_tree_with_cluster_label(threCluster_dict, table_names, dataset):
    # print(len(table_names))
    lowest_layer = []
    simple_tree = nx.DiGraph()
    target_path = f"datasets/{dataset}"
    table_gt = pd.read_csv(os.path.join(target_path, "groundTruth.csv"))
    for table in table_names:
        if table + ".csv" in list(table_gt['fileName']):
            labels = list(table_gt[table_gt['fileName'] == table + ".csv"]['class'])
        else:
            labels = []
        simple_tree.add_node(table, type='data', label=labels)
    clusterNodeId = len(table_names)
    last_layer_info = {}
    for index, (thre, clusters) in enumerate(threCluster_dict):
        layer_current = {}
        # print("the threshold is ", "{:.3f}".format(thre), "the cluster size is ", len(clusters), len(table_names))
        # print(f"current cluster {len(clusters), clusters}")
        """
        If index is 0, that means the lowest layer 
        """
        if index == 0:
            for cluster_id, tables_id in clusters.items():
                if len(tables_id) > 1:
                    tables = [table_names[ta] for ta in tables_id]
                    simple_tree.add_node(clusterNodeId, type='data cluster node')

                    updateNodeInfo(simple_tree, clusterNodeId, tables, dataset)
                    lowest_layer.append(clusterNodeId)

                    for ta in tables_id:
                        simple_tree.add_edge(clusterNodeId, table_names[ta])
                    layer_current[cluster_id] = clusterNodeId
                    clusterNodeId += 1
                else:
                    layer_current[cluster_id] = table_names[tables_id[0]]
                    lowest_layer.append(table_names[tables_id[0]])
        else:
            # print(f"last_layer_info {last_layer_info}")
            checked_ids = []
            for cluster_idx, tables_idx in clusters.items():
                left_tables = tables_idx
                for cluster_idz, tables_idz in threCluster_dict[index - 1][1].items():
                    current_idz = last_layer_info[cluster_idz]
                    if cluster_idz in checked_ids:
                        continue
                    else:
                        if tables_idx == tables_idz:
                            checked_ids.append(cluster_idz)
                            layer_current[cluster_idx] = current_idz
                            break
                        else:
                            is_subset = set(tables_idz).issubset(set(tables_idx))
                            if is_subset is True:
                                typeNode = 'data cluster node' if index != len(threCluster_dict) - 1 \
                                    else 'data cluster node parent layer'

                                # labelMode = 0
                                tables = [table_names[ta] for ta in tables_idx]
                                simple_tree.add_node(clusterNodeId, type=typeNode)
                                updateNodeInfo(simple_tree, clusterNodeId, tables, dataset)
                                simple_tree.add_edge(clusterNodeId, current_idz)
                                left_tables = [i for i in left_tables if i not in tables_idz]
                                checked_ids.append(cluster_idz)
                                if len(left_tables) == 0:
                                    layer_current[cluster_idx] = clusterNodeId
                                    clusterNodeId += 1
                                    break
                            else:
                                continue

        # check_nodes = [node for node in simple_tree.nodes() if simple_tree.in_degree(node) == 0]
        # print(f"check top level nodes{check_nodes}  \n")
        last_layer_info = layer_current

    Top_layer = [i for i in simple_tree.nodes() if simple_tree.in_degree(i) == 0]
    # PKL.hierarchy_tree(simple_tree)

    return simple_tree, lowest_layer, Top_layer


def find_most_similar_lists(base_list, other_lists):
    max_similarity = 0
    most_similar_lists = []
    for lst in other_lists:
        if isinstance(base_list[0], str):
            matched = [i for i in base_list if i in lst]
            if len(matched) > max_similarity:
                max_similarity = len(matched)
                most_similar_lists = lst
            return most_similar_lists, max_similarity
        elif isinstance(base_list[0], list):
            noIntersection = []
            for a in base_list:
                if not any(set(a).intersection(set(lst))):
                    noIntersection.append(a)
                if len(base_list) - len(noIntersection) > max_similarity:
                    max_similarity = len(base_list) - len(noIntersection)
                    most_similar_lists = [i for i in base_list if i not in noIntersection]
                return most_similar_lists, max_similarity


def TreeConsistencyScore(tree, lowest_layer, top_layer, dataset):
    overall_path_score = 0
    target_path = os.path.join(os.getcwd(), "datasets/" + dataset)
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)

    all_paths = []
    for bottom_node in lowest_layer:

        for top_node in top_layer:
            paths = list(nx.all_simple_paths(tree, top_node, bottom_node))
            all_paths.extend(paths)
    for path in all_paths:

        types_path = [tree.nodes[i]['label'] for i in path]
        has_path = False
        possible_paths = []
        bottom = types_path[-1]
        tops = types_path[0]
        special = False
        for top_node in tops:
            for bottom_node in bottom:
                if bottom_node == top_node:
                    has_path = True
                    special = True
                    continue
                else:
                    if nx.has_path(G, top_node, bottom_node):
                        paths = list(nx.all_simple_paths(G, top_node, bottom_node))
                        has_path = True
                        for subpath in paths:

                            possible_paths.append(subpath)
                    elif nx.has_path(G, bottom_node, top_node):
                        paths = list(nx.all_simple_paths(G, bottom_node, top_node))
                        has_path = True
                        for subpath in paths:

                            possible_paths.append(subpath)

        if has_path is False:
            #print("No path!", path, types_path, possible_paths)
            perConsistencyS = 0
        else:

            if special is True and possible_paths == []:
                #print("Direct!", types_path)
                perConsistencyS = 1

            else:
                matchedElement = len(types_path)

                most_similar_path, matchedElement_GT = find_most_similar_lists(types_path, possible_paths)

                perConsistencyS = matchedElement_GT / matchedElement
                #print("path types_path, possible paths", types_path, possible_paths, perConsistencyS)
        overall_path_score += perConsistencyS
    overall_path_score = overall_path_score / len(all_paths) if len(all_paths) > 0 else 1
    return overall_path_score, len(all_paths)


def tree_consistency_metric(cluster_name, tables, JaccardMatrix, embedding_file, dataset, Naming=None,
                            sliceInterval=10, delta=0.1, targetName=None):
    """data_path = os.path.join(os.getcwd(), "datasets", dataset, "groundTruth.csv")
    ground_truth_csv = pd.read_csv(data_path, encoding='latin1')"""
    encodings = [[i] for i in range(0, len(tables))]
    timing = {}
    layer_purity = []

    def custom_metric(index_table1, index_table2):
        score = 1
        table1 = tables[int(index_table1)]
        table2 = tables[int(index_table2)]
        # Replace this with your actual distance calculation logic
        if (table1, table2) in JaccardMatrix.keys():
            score = JaccardMatrix[(table1, table2)]
        elif (table2, table1) in JaccardMatrix.keys():
            score = JaccardMatrix[(table2, table1)]
        return score

    linkage_matrix = sch.linkage(encodings, method='single', metric=custom_metric)  # 'euclidean' # ward  complete

    # table_ids = [i for i in range(0, len(tables))]
    # plt.figure(figsize=(10, 7))
    dendrogra = sch.dendrogram(linkage_matrix, labels=tables)
    # plt.xticks(rotation=30)
    # plt.show()
    # tree_test = PKL.dendrogram_To_DirectedGraph(encodings, linkage_matrix, tables)
    # PKL.hierarchy_tree(tree_test)
    start_time = time.time()
    threCluster_dict = PKL.best_clusters(dendrogra, linkage_matrix, encodings,
                                         customMatrix=custom_metric, sliceInterval=sliceInterval, delta=delta)
    # print(threCluster_dict)
    end_time = time.time()
    # Calculate the elapsed time
    timing['Finding Layers'] = {'timing': end_time - start_time}
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")

    if threCluster_dict == []:  # len(threCluster_dict) == 1 and threCluster_dict[0][1] is None
        print("no hierarchy!")
        return 0, 0, None
    simple_tree, lower_layer, top_layer = \
        simple_tree_with_cluster_label(threCluster_dict, tables, dataset)

    # simple_tree = PKL.hierarchy_tree(simple_tree)
    # file_path = f"result/SILM/WDC/"
    # draw_interactive_graph(simple_tree,file_path=f"figure/SCT8_SBERT_HI/{targetName}.html")

    start_time = time.time()
    TCS, len_path = TreeConsistencyScore(simple_tree, lower_layer, top_layer, dataset)
    end_time = time.time()
    timing['Tree Consistency Score'] = {'timing': end_time - start_time}

    print(f"Total layer: {len(threCluster_dict)} TCS:  {TCS} #PATH is {len_path}")
    timing_df = pd.DataFrame(timing)
    # timing_df.to_csv(os.path.join(file_path, "timing.csv"))
    # print(timing_df)

    if Naming is not None:
        mkdir(f"result/SILM/{dataset}/{Naming}/{cluster_name}")
        result_folder = os.path.join("result/SILM/", dataset, Naming, cluster_name)
        file_path = os.path.join(result_folder, embedding_file, str(delta))
        mkdir(file_path)
        mkdir(result_folder)
        info_path = os.path.join(file_path, "all_info.csv")
        with open(os.path.join(file_path, cluster_name + "_results.pkl"), 'wb') as file:
            # Dump the data into the pickle file
            pickle.dump((dendrogra, linkage_matrix, threCluster_dict, simple_tree), file)
    return TCS, len_path, simple_tree


def hierarchicalColCluster(clustering, filename, embedding_file, Ground_t, hp: Namespace):
    # os.path.abspath(os.path.dirname(os.getcwd()))

    datafile_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                 "All/" + embedding_file + "/column")
    # ground_truth_table = os.getcwd() + "/datasets/TabFact/groundTruth.csv"
    data_path = os.getcwd() + "/datasets/%s/Test/" % hp.dataset
    # Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table)

    target_path = os.getcwd() + "/result/SILM/Column/" + \
                  hp.dataset + "/_gt_cluster.pickle"
    F_cluster = open(target_path, 'rb')
    KEYS = pickle.load(F_cluster)
    index_cols = int(filename.split("_")[0])
    # print("index col", index_cols, "\n")
    print(KEYS[index_cols])
    F_cluster = open(os.path.join(datafile_path, filename), 'rb')
    col_cluster = pickle.load(F_cluster)
    # print(col_cluster)

    tables = Ground_t[str(KEYS[index_cols])]
    # print(str(KEYS[index_cols]), tables )
    score_path = os.getcwd() + "/result/SILM/" + hp.dataset + "/" + f"{str(hp.delta)}/" + embedding_file + "/"
    # print(score_path)
    mkdir(score_path)
    if len(tables) > 1:
        jaccard_score = JaccardMatrix(col_cluster[clustering], data_path)[2]
        # print(jaccard_score)
        TCS, ALL_path, simple_tree = tree_consistency_metric(clustering, tables, jaccard_score, embedding_file,
                                                             hp.dataset,
                                                             str(index_cols), sliceInterval=hp.intervalSlice,
                                                             delta=hp.delta, targetName=str(index_cols))
        if 'TreeConsistencyScore.csv' in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'), index_col=0)
        else:
            df = pd.DataFrame(columns=['Top Level Entity', 'Tree Consistency Score', '#Paths', 'ClusteringAlgorithm'])
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))

        if str(index_cols) + clustering not in df.index:
            new_data = {'Top Level Entity': KEYS[index_cols], 'Tree Consistency Score': TCS, "#Paths": ALL_path,
                        'ClusteringAlgorithm': clustering}
            # print(new_data)
            new_row = pd.DataFrame([new_data], index=[str(index_cols) + clustering])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            # print(df)
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))
        else:
            df.loc[str(index_cols) + clustering, 'Top Level Entity'] = KEYS[index_cols]
            df.loc[str(index_cols) + clustering, 'Tree Consistency Score'] = TCS
            df.loc[str(index_cols) + clustering, '#Paths'] = ALL_path
            df.loc[str(index_cols) + clustering, 'ClusteringAlgorithm'] = clustering
            # print(df)
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))
