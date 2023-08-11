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

"""
# This is for drawing the paper example
ground_truth_csv = pd.read_csv(os.path.join(os.getcwd(), "datasets/TabFact/Try.csv"))
#contains_sport = ground_truth_csv['LowestClass'].str.contains('sport', case=False, na=False)
contains_sport = ground_truth_csv["LowestClass"].str.contains("sport", case=False, na=False) & \
            (ground_truth_csv["LowestClass"].str.contains("competition", case=False, na=False) |
             ground_truth_csv["LowestClass"].str.contains("league", case=False, na=False) |
             ground_truth_csv["LowestClass"].str.contains("team", case=False, na=False))
result_df = ground_truth_csv[contains_sport]

print(result_df['LowestClass'].unique(), len(result_df['LowestClass'].unique()))

labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))

G = nx.DiGraph()
len_G = 0

for index, row in result_df.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:7]
        all_nodes = set(df.values.ravel())
        if len(set(G.nodes())):
            all_nodes = all_nodes - set(G.nodes())
        else:
            nodes_set = all_nodes
        G.add_nodes_from(all_nodes)
        nodes_set = set(G.nodes())
        for _, row2 in df.iterrows():
            labels_table = row2.dropna().tolist()
            for i in range(len(labels_table) - 1):
                G.add_edge(labels_table[i + 1], labels_table[i])

    else:
        if row["class"]!=" ":
            superclass = row["superclass"]
            classX = row["class"]
            all_nodes = {superclass, classX}
            all_nodes = all_nodes - set(G.nodes())
            if len(all_nodes) > 0:

                G.add_nodes_from(all_nodes)
                G.add_edge(superclass, classX)
    if len_G<len(set(G.nodes())):
        graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
        plt.show()
"""


def find_frequent_labels(ancestors: list, G: nx.DiGraph()):
    indegrees = [G.in_degree(parent) for parent in ancestors]
    min_element = min(indegrees)
    min_indices = [i for i, x in enumerate(indegrees) if x == min_element]
    topmost_parent = [ancestors[i] for i in min_indices]
    return topmost_parent


def labels_most_fre(datas: dict):
    if len(datas) == 1:
        return list(datas.values())[0]
    # frequency of each label
    label_counter = Counter(label for labels_list in datas.values() for label in labels_list)
    # Mayjor voting label
    most_common_labels = label_counter.most_common()

    # most frequently appeared label
    max_frequency = most_common_labels[0][1]
    most_common_labels = [label for label, frequency in most_common_labels if frequency == max_frequency]

    return most_common_labels


def ground_truth_labels(filename, top=False):
    ground_label_name1 = "01SourceTables.csv"  # "GroundTruth.csv"
    data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
    target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)
    row = ground_truth_csv[ground_truth_csv['fileName'] == filename]
    All_ancestors = []
    topmost_parent = []
    classX = row["class"].iloc[0]  # LowestClass
    # table_label = ast.literal_eval(classX)

    """for loa in table_label:

        ancestors = list(nx.ancestors(G, loa)) if len(list(nx.ancestors(G, loa)))>0 else [loa]
        for lax in ancestors:
            All_ancestors.append(lax)
        freq_final = find_frequent_labels(ancestors, G)
        for la in freq_final:
            topmost_parent.append(la)"""
    labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
    if filename in labels:
        table_label = pd.read_csv(
            os.path.join(os.getcwd(), "datasets/TabFact/Label", filename))[
            "classLabel"].unique()
        for loa in table_label:

            ancestors = list(nx.ancestors(G, loa)) if len(list(nx.ancestors(G, loa))) > 0 else [loa]
            for lax in ancestors:
                All_ancestors.append(lax)
            freq_final = find_frequent_labels(ancestors, G)
            for la in freq_final:
                topmost_parent.append(la)
    else:
        ancestors = list(nx.ancestors(G, classX))
        All_ancestors = list(nx.ancestors(G, classX))
        if len(ancestors) == 0:
            topmost_parent = [classX]
        else:
            topmost_parent = find_frequent_labels(ancestors, G)

    if top is True:
        return topmost_parent
    else:
        All_ancestors = [item for item in All_ancestors if item not in topmost_parent]
        if len(All_ancestors) == 0:
            All_ancestors = topmost_parent
        return All_ancestors


def label_dict(tables: list, is_Parent=True):
    dict_table_labels = {}
    for table in tables:
        dict_table_labels[table] = ground_truth_labels(table + ".csv", top=is_Parent)
    return dict_table_labels


def no_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    if len(set1.intersection(set2)) == 0:
        return True
    else:
        return False


def simple_tree_with_cluster_label(threCluster_dict, orginal_tree, table_names, timing):
    layer_info_dict = {}
    Parent_nodes_h = {}
    simple_tree = None
    for index, (thre, clusters) in enumerate(threCluster_dict):
        print("{:.3f}".format(thre), len(clusters))
    for index, (thre, clusters) in enumerate(threCluster_dict):
        # print(thre, len(clusters))
        if index == 0:
            start_time = time.time()
            parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
            end_time = time.time()
            # Calculate the elapsed time
            # elapsed_time = end_time - start_time
            # print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the best silhouette tree")

            # print(parent_nodes)
            simple_tree = PKL.simplify_graph(orginal_tree, parent_nodes)
            # print(simple_tree)
        start_time = time.time()
        parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
        layer_info_dict[index] = parent_nodes
        end_time = time.time()
        # Calculate the elapsed time
        # elapsed_time = end_time - start_time
        timing[str(index) + '_slicing'] = {'timing': end_time - start_time}
        # print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the tree")
        layer_num = len(threCluster_dict) - 1
        start_time_label = time.time()

        for cluster, (closest_parent, mutual_parent_nodes) in parent_nodes.items():

            simple_tree.nodes[closest_parent]['data'] = list(cluster)
            if index == layer_num:
                label_dict_cluster = label_dict(list(cluster), is_Parent=True)
                simple_tree.nodes[closest_parent]['label'] = labels_most_fre(label_dict_cluster)
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node parent layer'
                Parent_nodes_h[cluster] = closest_parent
            else:
                label_dict_cluster = label_dict(list(cluster), is_Parent=False)
                simple_tree.nodes[closest_parent]['label'] = labels_most_fre(label_dict_cluster)
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node'
            wrong_labels = {}
            # print(simple_tree.nodes[closest_parent]['label'])
            for i in cluster:
                if no_intersection(label_dict_cluster[i], simple_tree.nodes[closest_parent]['label']) is True:
                    wrong_labels[i] = label_dict_cluster[i]
            simple_tree.nodes[closest_parent]['Wrong_labels'] = wrong_labels
            label_cluster = simple_tree.nodes[closest_parent]['label']
            simple_tree.nodes[closest_parent]['Purity'] = "{:.3f}".format(
                (1 - len(wrong_labels.keys()) / len(cluster)) / len(label_cluster))
        end_time_label = time.time()
        timing[str(index) + '_label'] = {'timing': end_time_label - start_time_label}
        # elapsed_time = end_time_label - start_time_label
        # print(f"Elapsed time: {elapsed_time:.4f} seconds for calculate the label of the tree")
    return simple_tree, layer_info_dict, Parent_nodes_h


def TreeConsistencyScore(tree, layer_info, Parent_nodes, strict=False):
    target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
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
                        # print(label)
                        if label != 0:
                            """print(f" Node: {i}",f" Type: {tree.nodes[i].get('type', 0)} \n",
                                  f" cluster label: {label} \n",
                                  f" Purity: {tree.nodes[i].get('Purity', 0)} \n"
                                  f" Data: {tree.nodes[i].get('data', 0)} \n",
                                  f" Wrong Labels: {tree.nodes[i].get('data', 0)} \n",
                                  )"""
                            if superLabels != 0:
                                topmost_parent = []
                                for label_per in label:
                                    ancestors = list(nx.ancestors(G, label_per))
                                    if len(ancestors) == 0:
                                        freq_final = [label_per]
                                    else:
                                        freq_final = find_frequent_labels(ancestors, G)
                                    for la in freq_final:
                                        topmost_parent.append(la)

                                    for superLabel in topmost_parent:
                                        if superLabel in superLabels:
                                            MatchedElementGT += 1
                                            break  # Break out of the inner loop when the condition is met
                                    else:
                                        continue

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

    linkage_matrix = sch.linkage(encodings, method='complete', metric=custom_metric)  # 'euclidean'
    # print(linkage_matrix)
    result_folder = os.path.join("result/Valerie/", dataset, Naming, cluster_name)
    mkdir(result_folder)
    dendrogra = sch.dendrogram(linkage_matrix, labels=tables)
    # dendrogra = PKL.plot_tree(linkage_matrix, result_folder, node_labels=tables)
    tree_test = PKL.dendrogram_To_DirectedGraph(encodings, linkage_matrix, tables)
    start_time = time.time()
    layers = 3
    threCluster_dict = PKL.best_clusters(custom_metric, dendrogra, linkage_matrix, encodings,
                                         estimate_num_cluster=layers)
    end_time = time.time()
    # Calculate the elapsed time
    timing['Finding Layers'] = {'timing': end_time - start_time}
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")
    if len(threCluster_dict) == 1 and threCluster_dict[0][1] is None:
        print("no hierarchy!")
        return None
    simple_tree, layer_info_dict, Parent_nodes_h = \
        simple_tree_with_cluster_label(threCluster_dict, tree_test, tables, timing)
    start_time = time.time()
    TCS, len_path = TreeConsistencyScore(simple_tree, layer_info_dict[0], Parent_nodes_h)
    end_time = time.time()
    timing['Tree Consistency Score'] = {'timing': end_time - start_time}
    # PKL.print_clusters_top_down(simple_tree, layer_info_dict)
    Purity_layers = PKL.purity_per_layer(simple_tree, layer_info_dict)
    for layer, purity_list in Purity_layers.items():
        purity_list = [float(i) for i in purity_list]
        Purity_layer = "{:.3f}".format(sum(purity_list) / len(purity_list))
        # print(f"layer {layer} purity is: {Purity_layer} number is {len(purity_list)}")
        layer_purity.append({'layer': layer, 'nodes NO': len(purity_list), 'Purity': Purity_layer})
    file_path = os.path.join(result_folder, embedding_file)
    mkdir(file_path)
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
    return TCS


def hierarchicalColCluster(clustering, filename, hp: Namespace):
    # os.path.abspath(os.path.dirname(os.getcwd()))
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    embedding_file = [fn for fn in os.listdir(datafile_path)
                      if fn.endswith(hp.embedMethod + '.pkl') and hp.embed in fn][0][0:-4]
    datafile_path = os.path.join(os.getcwd(), "result/starmie/", hp.dataset,
                                 "All/" + embedding_file + "/column")
    ground_truth_table = os.getcwd() + "/datasets/TabFact/groundTruth.csv"
    data_path = os.getcwd() + "/datasets/TabFact/Test/"
    Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table)

    target_path = os.getcwd() + "/result/Valerie/Column/" + \
                  hp.dataset + "/_gt_cluster.pickle"
    F_cluster = open(target_path, 'rb')
    KEYS = pickle.load(F_cluster)

    index_cols = int(filename.split("_")[0])
    F_cluster = open(os.path.join(datafile_path, filename), 'rb')
    col_cluster = pickle.load(F_cluster)
    tables = Ground_t[KEYS[index_cols]]
    score_path = os.getcwd() + "/result/Valerie/"+ hp.dataset + "/" + embedding_file + "/"
    print(score_path)
    mkdir(score_path)
    if len(tables) > 1:
        jaccard_score = JaccardMatrix(col_cluster[clustering], data_path)[2]
        TCS = tree_consistency_metric(clustering, tables, jaccard_score, embedding_file, hp.dataset,
                                      str(index_cols))
        if 'TreeConsistencyScore.csv' in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'), index_col=0)
        else:
            df = pd.DataFrame(columns=['Top Level Entity', 'Tree Consistency Score'])
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))

        if index_cols not in df.index:
            new_data = {'Top Level Entity': KEYS[index_cols], 'Tree Consistency Score': TCS}
            new_row = pd.DataFrame([new_data], index=[index_cols])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            df.to_csv(os.path.join(score_path, 'TreeConsistencyScore.csv'))
