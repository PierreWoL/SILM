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


def ground_truth_labels(filename, mode=0, dataset="TabFact"):
    if mode > 2:
        print("wrong label mode!")
        return None
    ground_label_name1 = "groundTruth.csv"  #
    data_path = os.path.join(os.getcwd(), "datasets/" + dataset, ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
    target_path = os.path.join(os.getcwd(), "datasets/" + dataset)
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)

    top_nodes = [i for i in G.nodes() if G.in_degree(i) == 0]
    successors = [succ for node in top_nodes for succ in G.successors(node)]

    row = ground_truth_csv[ground_truth_csv['fileName'] == filename]

    classX = [row["class"].iloc[0]] if dataset != "TabFact" else ast.literal_eval(row["class"].iloc[0])  # LowestClass

    if mode == 0:
        return classX
    elif mode == 1:
        # Return the intermedia type of this cluster
        all_anc = []
        for class_file in classX:
            ancestors = [i for i in nx.ancestors(G, class_file)
                         if i not in successors and i not in top_nodes and G.out_degree(i) != 0]
            all_anc.extend(ancestors)

        if len(all_anc) == 0:
            all_anc = classX

        return all_anc

    elif mode == 2:
        # Return the top level type of this cluster
        all_anc = []
        for class_file in classX:
            ancestors = [i for i in nx.ancestors(G, class_file) if i in successors]
            all_anc.extend(ancestors)
        if len(all_anc) == 0:
            all_anc = classX

        return all_anc


def label_dict(tables: list, mode=0, dataset="TabFact"):
    dict_table_labels = {}

    for table in tables:
        dict_table_labels[table] = ground_truth_labels(table + ".csv", mode=mode, dataset=dataset)

    return dict_table_labels


def no_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if len(set1.intersection(set2)) == 0:
        return True
    else:
        return False


def updateNodeInfo(tree, clusterNode, tables, dataset, mode=0):
    label_dict_cluster = label_dict(list(tables), mode=mode, dataset=dataset)  # True
    labels, freq = labels_most_fre(label_dict_cluster)
    tree.nodes[clusterNode]['label'] = labels
    wrong_labels = {}
    for i in tables:
        if no_intersection(label_dict_cluster[i], tree.nodes[clusterNode]['label']) is True:
            # this have problems
            wrong_labels[i] = label_dict_cluster[i]
    tree.nodes[clusterNode]['Wrong_labels'] = wrong_labels
    tree.nodes[clusterNode]['Purity'] = freq / len(tables)


def simple_tree_with_cluster_label(threCluster_dict, table_names, dataset):
    # print(len(table_names))
    lowest_layer = []

    simple_tree = nx.DiGraph()
    for table in table_names:
        label_dict_cluster = label_dict([table], mode=0, dataset=dataset)  # True
        labels, freq = labels_most_fre(label_dict_cluster)
        simple_tree.add_node(table, type='data', label=labels)
    clusterNodeId = len(table_names)
    last_layer_info = {}
    for index, (thre, clusters) in enumerate(threCluster_dict):
        layer_current = {}
        # print("the threshold is ", "{:.3f}".format(thre), "the cluster size is ", len(clusters))
        # print(f"current cluster {len(clusters), clusters}")
        if index == 0:
            for cluster_id, tables_id in clusters.items():
                if len(tables_id) > 1:
                    tables = [table_names[ta] for ta in tables_id]
                    simple_tree.add_node(clusterNodeId, type='data cluster node')
                    updateNodeInfo(simple_tree, clusterNodeId, tables, dataset, mode=0)
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
                                labelMode = 0 if index != len(threCluster_dict) - 1 else 2
                                tables = [table_names[ta] for ta in tables_idx]
                                simple_tree.add_node(clusterNodeId, type=typeNode)
                                updateNodeInfo(simple_tree, clusterNodeId, tables, dataset, mode=labelMode)
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


"""
def TreeConsistencyScore(tree, layer_info, Parent_nodes, strict=True, dataset="TabFact"):
    target_path = os.path.join(os.getcwd(), "datasets/" + dataset)
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)
    all_pathCompatibility = []

    ind, ind_o = 0, 0
    for cluster, (closest_parent, mutual_parent_nodes) in layer_info.items():
        path = sorted(list(mutual_parent_nodes))
        ind_o += 1
        for indexp, (clusterP, closest_parentP) in enumerate(Parent_nodes.items()):
            if closest_parentP in path:
                superLabels = tree.nodes[closest_parentP].get('label', 0)
                path = path[0:path.index(closest_parentP)]
                MatchedElementPath = len(path)
                MatchedElementGT = 0
                if len(path) > 0:
                    ind += 1
                    for index, i in enumerate(path):
                        label = tree.nodes[i].get('label', 0)
                        if label != 0:
                           
                           # print(f" Node: {i}", f" Type: {tree.nodes[i].get('type', 0)} \n",
                             #     f" cluster label: {label} \n",
                             #     f" Purity: {tree.nodes[i].get('Purity', 0)} \n"
                            #      f" Data: {tree.nodes[i].get('data', 0)} \n")
                                  
                            if superLabels != 0:
                                if strict is False:
                                    topmost_parent = []
                                    for label_per in label:
                                        ancestors = list(nx.ancestors(G, label_per))
                                        if len(ancestors) == 0:
                                            freq_final = [label_per]
                                        else:
                                            freq_final = find_frequent_labels(ancestors, G)[0]
                                        for la in freq_final:
                                            topmost_parent.append(la)
                                        # print(F'topmost_parent label of {label_per} is {topmost_parent}')

                                        for superLabel in topmost_parent:
                                            if superLabel in superLabels:
                                                MatchedElementGT += 1
                                                break  # Break out of the inner loop when the condition is met
                                        else:
                                            continue
                                        break
                                else:
                                    for label_per in label:
                                        if label_per in superLabels:
                                            MatchedElementGT += 1
                                            break
                        else:
                            MatchedElementPath -= 1
                        if index == 0 and MatchedElementGT == 0:
                            break
                    if MatchedElementPath != 0:
                        pathCompatibility = MatchedElementGT / MatchedElementPath
                        all_pathCompatibility.append(pathCompatibility)
    if len(all_pathCompatibility) != 0:
        tree_consistency_score = "{:.3f}".format(sum(all_pathCompatibility) / len(all_pathCompatibility))
    else:
        tree_consistency_score = None

    return tree_consistency_score, len(all_pathCompatibility)
"""


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
        matchedElement_GT = 0
        matchedElement = 0
        types_path = [tree.nodes[i]['label'] for i in path]
        has_path = False
        intersection = set(types_path[0]).intersection(set(types_path[-1]))
        if intersection:
            matchedElement = len(types_path)
            for index, types in enumerate(types_path):
                if index == 0 or index == len(types_path) - 1:
                    matchedElement_GT += 1

                else:
                    intersection_other = set(types_path[0]).intersection(set(types))
                    if intersection_other:
                        matchedElement_GT += 1
            perConsistencyS = matchedElement_GT / matchedElement

        else:
            possible_paths_elements = set()
            for top_node in types_path[0]:
                for bottom_node in types_path[-1]:
                    if nx.has_path(G, top_node, bottom_node):
                        paths = list(nx.all_simple_paths(G, top_node, bottom_node))
                        has_path = True
                        matchedElement = len(types_path)
                        for sublist in paths:
                            possible_paths_elements.update(sublist)
                    elif nx.has_path(G, bottom_node, top_node):
                        paths = list(nx.all_simple_paths(G, bottom_node, top_node))
                        has_path = True
                        matchedElement = len(types_path)
                        for sublist in paths:
                            possible_paths_elements.update(sublist)
            if has_path is False:
                perConsistencyS = 0
                # print(types_path, perConsistencyS)
            else:
                for index, types in enumerate(types_path):
                    if index == 0 or index == len(types_path) - 1:
                        matchedElement_GT += 1

                    else:
                        intersection_other = set(possible_paths_elements).intersection(set(types))
                        if intersection_other:
                            matchedElement_GT += 1
                perConsistencyS = matchedElement_GT / matchedElement

        overall_path_score += perConsistencyS
    overall_path_score = overall_path_score / len(all_paths) if len(all_paths) > 0 else 1
    return overall_path_score, len(all_paths)


def tree_consistency_metric(cluster_name, tables, JaccardMatrix, embedding_file, dataset, Naming,
                            sliceInterval=10, delta=0.1):
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

    result_folder = os.path.join("result/SILM/", dataset, Naming, cluster_name)
    file_path = os.path.join(result_folder, embedding_file)
    mkdir(file_path)
    mkdir(f"result/SILM/{dataset}/{Naming}/{cluster_name}")
    linkage_matrix = sch.linkage(encodings, method='complete', metric=custom_metric)  # 'euclidean'

    mkdir(result_folder)
    # table_ids = [i for i in range(0, len(tables))]
    # plt.figure(figsize=(10, 7))
    dendrogra = sch.dendrogram(linkage_matrix, labels=tables)
    plt.xticks(rotation=30)
    plt.show()
    # tree_test = PKL.dendrogram_To_DirectedGraph(encodings, linkage_matrix, tables)
    start_time = time.time()
    # layers = 4
    threCluster_dict = PKL.best_clusters(dendrogra, linkage_matrix, encodings,
                                         customMatrix=custom_metric, sliceInterval=sliceInterval, delta=delta)
    #print(threCluster_dict)
    end_time = time.time()
    # Calculate the elapsed time
    timing['Finding Layers'] = {'timing': end_time - start_time}
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")
    if threCluster_dict ==[]:#len(threCluster_dict) == 1 and threCluster_dict[0][1] is None
        print("no hierarchy!")
        return  0,0
    simple_tree, lower_layer, top_layer = \
        simple_tree_with_cluster_label(threCluster_dict, tables, dataset)
    start_time = time.time()
    TCS, len_path = TreeConsistencyScore(simple_tree, lower_layer, top_layer, dataset)
    end_time = time.time()
    timing['Tree Consistency Score'] = {'timing': end_time - start_time}
    info_path = os.path.join(file_path, "all_info.csv")
    print(f"Total layer: {len(threCluster_dict)} TCS:  {TCS} #PATH is {len_path}")
    timing_df = pd.DataFrame(timing)
    # timing_df.to_csv(os.path.join(file_path, "timing.csv"))
    # print(timing_df)
    with open(os.path.join(file_path, cluster_name + "_results.pkl"), 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump((dendrogra, linkage_matrix, threCluster_dict, simple_tree), file)
    return TCS, len_path


def hierarchicalColCluster(clustering, filename, embedding_file, Ground_t, hp: Namespace):
    # os.path.abspath(os.path.dirname(os.getcwd()))
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"

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

    tables = Ground_t[str(KEYS[index_cols])]
    print(str(KEYS[index_cols]), tables )
    score_path = os.getcwd() + "/result/SILM/" + hp.dataset + "/" + embedding_file + "/"
    # print(score_path)
    mkdir(score_path)
    if len(tables) > 1:
        jaccard_score = JaccardMatrix(col_cluster[clustering], data_path)[2]
        #print(jaccard_score)
        TCS, ALL_path = tree_consistency_metric(clustering, tables, jaccard_score, embedding_file, hp.dataset,
                                                str(index_cols), sliceInterval= hp.intervalSlice, delta=hp.delta)
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


"""
def simple_tree_with_cluster_label(threCluster_dict, orginal_tree, table_names, timing, dataset):
    layer_info_dict = {}
    Parent_nodes_h = {}
    simple_tree = None
    lowerNodes = None
    for index, (thre, clusters) in enumerate(threCluster_dict):
        print("the threshold is ", "{:.3f}".format(thre), "the cluster size is ", len(clusters))
    for index, (thre, clusters) in enumerate(threCluster_dict):
        print("check here", index)
        if index == 0:
            parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
            simple_tree = PKL.simplify_graph(orginal_tree, parent_nodes)
        else:
            parent_nodes = PKL.slice_tree(simple_tree, clusters, table_names)  #
            simple_tree = PKL.simplify_graph(simple_tree, parent_nodes, lowerNode=lowerNodes)
        if index == len(threCluster_dict) - 1:
            top_node = [i for i in simple_tree.nodes() if simple_tree.in_degree(i) == 0][0]
            clusterDict = {0: (top_node, {top_node})}
            curentNodes = set([i[0] for i in list(parent_nodes.values())])
            simple_tree = PKL.simplify_graph(simple_tree, clusterDict, lowerNode=curentNodes)

        PKL.hierarchy_tree(simple_tree)
        parent_nodes = PKL.slice_tree(simple_tree, clusters, table_names)
        #print(parent_nodes)
        lowerNodes = set([i[0] for i in list(parent_nodes.values())])
       # print(lowerNodes)
        # PKL.hierarchy_tree(simple_tree)
        start_time = time.time()
        # print(parent_nodes,"\n")

        layer_info_dict[index] = parent_nodes
        end_time = time.time()
        # Calculate the elapsed time
        timing[str(index) + '_slicing'] = {'timing': end_time - start_time}
        # print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the tree")
        layer_num = len(threCluster_dict) - 1
        start_time_label = time.time()
        for cluster, (closest_parent, mutual_parent_nodes) in parent_nodes.items():
            simple_tree.nodes[closest_parent]['data'] = list(cluster)
            if index == layer_num:
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node parent layer'
                Parent_nodes_h[cluster] = closest_parent
            else:
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node'

            label_dict_cluster = label_dict(list(cluster), mode=1, dataset=dataset)  # True

            labels, freq = labels_most_fre(label_dict_cluster)

            simple_tree.nodes[closest_parent]['label'] = labels
            wrong_labels = {}
            for i in cluster:
                if no_intersection(label_dict_cluster[i], simple_tree.nodes[closest_parent]['label']) is True:
                    # this have problems
                    wrong_labels[i] = label_dict_cluster[i]
            simple_tree.nodes[closest_parent]['Wrong_labels'] = wrong_labels
            label_cluster = simple_tree.nodes[closest_parent]['label']
            simple_tree.nodes[closest_parent]['Purity'] = freq / len(cluster)

        end_time_label = time.time()
        timing[str(index) + '_label'] = {'timing': end_time_label - start_time_label}
        # elapsed_time = end_time_label - start_time_label
        # print(f"Elapsed time: {elapsed_time:.4f} seconds for calculate the label of the tree") 

    return simple_tree, layer_info_dict, Parent_nodes_h

"""
