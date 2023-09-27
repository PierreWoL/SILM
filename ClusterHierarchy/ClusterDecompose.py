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


def ground_truth_labels(filename, top=False, inter=False,dataset="TabFact"):
    ground_label_name1 ="GroundTruth.csv" #
    data_path = os.path.join(os.getcwd(), "datasets/"+dataset, ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
    target_path = os.path.join(os.getcwd(), "datasets/"+dataset)
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)
    row = ground_truth_csv[ground_truth_csv['fileName'] == filename]
    lowest_parent = []
    All_ancestors = []
    topmost_parent = []
    classX = row["class"].iloc[0]  # LowestClass
    # table_label = ast.literal_eval(classX)
    if os.path.exists(os.path.join(os.getcwd(), "datasets/%s/Label" %dataset)):

        labels = os.listdir(os.path.join(os.getcwd(), "datasets/%s/Label" %dataset))
    else:
        labels=[]
    if filename in labels:
        table_label = pd.read_csv(
            os.path.join(os.getcwd(), "datasets/%s/Label"%dataset, filename))[
            "classLabel"].unique()
        for loa in table_label:
            lowest_parent.append(loa)
            if top is True:
                ancestors = list(nx.ancestors(G, loa)) if len(list(nx.ancestors(G, loa))) > 0 else [loa]
                for lax in ancestors:
                    All_ancestors.append(lax)
                freq_final = find_frequent_labels(ancestors, G)[0]
                for la in freq_final:
                    topmost_parent.append(la)
    else:
        lowest_parent.append(classX)
        if top is True:
            ancestors = list(nx.ancestors(G, classX))
            All_ancestors = list(nx.ancestors(G, classX))
            if len(ancestors) == 0:
                topmost_parent = [classX]
            else:
                topmost_parent = find_frequent_labels(ancestors, G)[0]

    if top is True:
        return topmost_parent
    if inter is True:
        # All ancestors is the parent conceptual type but not in the top-level
        All_ancestors = [item for item in All_ancestors if item not in topmost_parent]
        if len(All_ancestors) == 0:
            All_ancestors = topmost_parent
        return All_ancestors

    return [classX]


def label_dict(tables: list, is_Parent=True,dataset = "TabFact"):
    dict_table_labels = {}
    for table in tables:
        dict_table_labels[table] = ground_truth_labels(table + ".csv", top=is_Parent,dataset=dataset)
    return dict_table_labels


def no_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if len(set1.intersection(set2)) == 0:
        return True
    else:
        return False


def simple_tree_with_cluster_label(threCluster_dict, orginal_tree, table_names, timing,dataset):
    layer_info_dict = {}
    Parent_nodes_h = {}
    simple_tree = None
    for index, (thre, clusters) in enumerate(threCluster_dict):
        print("{:.3f}".format(thre), len(clusters))
    for index, (thre, clusters) in enumerate(threCluster_dict):
        if index == 0:
            start_time = time.time()
            parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
            end_time = time.time()
            # print(f'parent_nodes {parent_nodes}')
            simple_tree = PKL.simplify_graph(orginal_tree, parent_nodes)

        start_time = time.time()
        parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
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
            label_dict_cluster = label_dict(list(cluster), is_Parent=False,dataset=dataset)  # True
            labels, freq = labels_most_fre(label_dict_cluster)
            simple_tree.nodes[closest_parent]['label'] = labels
            """if len(cluster) > 1:
                    print(f' node {closest_parent} is {label_dict_cluster}',
                          simple_tree.nodes[closest_parent]['label'])"""

            wrong_labels = {}
            # print(simple_tree.nodes[closest_parent]['label'])
            for i in cluster:
                if no_intersection(label_dict_cluster[i], simple_tree.nodes[closest_parent]['label']) is True:
                    # this have problems
                    wrong_labels[i] = label_dict_cluster[i]
            simple_tree.nodes[closest_parent]['Wrong_labels'] = wrong_labels
            label_cluster = simple_tree.nodes[closest_parent]['label']
            """ 
            if len(label_cluster)>1:
                simple_tree.nodes[closest_parent]['Purity'] = freq/ len(cluster)
            else:
                simple_tree.nodes[closest_parent]['Purity'] = 1 - len(wrong_labels) / len(cluster)
            """
            simple_tree.nodes[closest_parent]['Purity'] = freq / len(cluster)

        end_time_label = time.time()
        timing[str(index) + '_label'] = {'timing': end_time_label - start_time_label}
        # elapsed_time = end_time_label - start_time_label
        # print(f"Elapsed time: {elapsed_time:.4f} seconds for calculate the label of the tree")
    return simple_tree, layer_info_dict, Parent_nodes_h


def TreeConsistencyScore(tree, layer_info, Parent_nodes, strict=True, dataset = "TabFact"):
    target_path = os.path.join(os.getcwd(), "datasets/"+dataset)
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)
    all_pathCompatibility = []
    # print(Parent_nodes)
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
                            """  
                            print(f" Node: {i}", f" Type: {tree.nodes[i].get('type', 0)} \n",
                                  f" cluster label: {label} \n",
                                  f" Purity: {tree.nodes[i].get('Purity', 0)} \n"
                                  f" Data: {tree.nodes[i].get('data', 0)} \n")
                                  """
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
                                        print(F'topmost_parent label of {label_per} is {topmost_parent}')

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


def tree_consistency_metric(cluster_name, tables, JaccardMatrix, embedding_file, dataset, Naming):
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
        # print(table1, table2, score)
        return score

    result_folder = os.path.join("result/Valerie/", dataset, Naming, cluster_name)
    file_path = os.path.join(result_folder, embedding_file)
    mkdir(file_path)
    linkage_matrix = sch.linkage(encodings, method='complete', metric=custom_metric)  # 'euclidean'

    mkdir(result_folder)
    table_ids = [i for i in range(0, len(tables))]
    dendrogra = sch.dendrogram(linkage_matrix, labels=tables)
    # dendrogra = PKL.plot_tree(linkage_matrix, file_path, node_labels=tables)
    tree_test = PKL.dendrogram_To_DirectedGraph(encodings, linkage_matrix, tables)
    start_time = time.time()
    layers = 4
    threCluster_dict = PKL.best_clusters(dendrogra, linkage_matrix, encodings,
                                         estimate_num_cluster=layers, customMatrix=custom_metric)
    end_time = time.time()
    # Calculate the elapsed time
    timing['Finding Layers'] = {'timing': end_time - start_time}
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")
    if len(threCluster_dict) == 1 and threCluster_dict[0][1] is None:
        print("no hierarchy!")
        return None
    simple_tree, layer_info_dict, Parent_nodes_h = \
        simple_tree_with_cluster_label(threCluster_dict, tree_test, tables, timing,dataset)
    start_time = time.time()
    TCS, len_path = TreeConsistencyScore(simple_tree, layer_info_dict[0], Parent_nodes_h,dataset=dataset)
    end_time = time.time()
    timing['Tree Consistency Score'] = {'timing': end_time - start_time}
    info_path = os.path.join(file_path, "all_info.csv")
    PKL.print_clusters_top_down(simple_tree, layer_info_dict, store_path=info_path)
    Purity_layers = PKL.purity_per_layer(simple_tree, layer_info_dict)
    for layer, purity_list in Purity_layers.items():
        purity_list = [float(i) for i in purity_list]
        Purity_layer = "{:.3f}".format(sum(purity_list) / len(purity_list))
        print(f"layer {layer} purity is: {Purity_layer} number is {len(purity_list)}")
        layer_purity.append({'layer': layer, 'nodes NO': len(purity_list), 'Purity': Purity_layer})

    print("Tree consistency metric: ", TCS, "all_path_number is ", len_path)
    layer_purity_df = pd.DataFrame(layer_purity)
    timing_df = pd.DataFrame(timing)
    layer_purity_df.to_csv(os.path.join(file_path, "layer_purity.csv"), index=False)
    timing_df.to_csv(os.path.join(file_path, "timing.csv"))
    # print(layer_purity_df)
    # print(timing_df)
    with open(os.path.join(file_path, cluster_name + "_results.pkl"), 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump((dendrogra, linkage_matrix, threCluster_dict,
                     simple_tree, tree_test, layer_info_dict, Parent_nodes_h), file)
    return TCS,len_path


def hierarchicalColCluster(clustering, filename,embedding_file, Ground_t,hp: Namespace):
    # os.path.abspath(os.path.dirname(os.getcwd()))
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"

    datafile_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                 "All/" + embedding_file + "/column")
    # ground_truth_table = os.getcwd() + "/datasets/TabFact/groundTruth.csv"
    data_path = os.getcwd() + "/datasets/%s/Test/" %hp.dataset
    # Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table)

    target_path = os.getcwd() + "/result/Valerie/Column/" + \
                  hp.dataset + "/_gt_cluster.pickle"
    F_cluster = open(target_path, 'rb')
    KEYS = pickle.load(F_cluster)
    index_cols = int(filename.split("_")[0])
    print(index_cols,"\n")
    print(KEYS[index_cols])
    F_cluster = open(os.path.join(datafile_path, filename), 'rb')
    col_cluster = pickle.load(F_cluster)
    tables = Ground_t[str(KEYS[index_cols])]
    score_path = os.getcwd() + "/result/Valerie/" + hp.dataset + "/" + embedding_file + "/"
    #print(score_path)
    mkdir(score_path)
    if len(tables) > 1:
        jaccard_score = JaccardMatrix(col_cluster[clustering], data_path)[2]
        TCS,ALL_path = tree_consistency_metric(clustering, tables, jaccard_score, embedding_file, hp.dataset,
                                      str(index_cols))
        if 'TreeConsistencyScore.csv' in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'), index_col=0)
        else:
            df = pd.DataFrame(columns=['Top Level Entity', 'Tree Consistency Score','#Paths', 'ClusteringAlgorithm'])
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))

        if str(index_cols)+clustering not in df.index:
            new_data = {'Top Level Entity': KEYS[index_cols], 'Tree Consistency Score': TCS, "#Paths":ALL_path,'ClusteringAlgorithm':clustering}
            print(new_data)
            new_row = pd.DataFrame([new_data], index=[str(index_cols)+clustering])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))
        else:
            df.loc[str(index_cols)+clustering,'Top Level Entity'] = KEYS[index_cols]
            df.loc[str(index_cols)+clustering, 'Tree Consistency Score'] =TCS
            df.loc[str(index_cols)+clustering, '#Paths'] =ALL_path
            df.loc[str(index_cols)+clustering, 'ClusteringAlgorithm'] = clustering
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))

