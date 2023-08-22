import pickle
import time

import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import readPKL as PKL
import scipy.cluster.hierarchy as sch

from Utils import mkdir

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
print(len(result_df))
for index, row in result_df.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
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
                # print(all_nodes)
                G.add_nodes_from(all_nodes)
                G.add_edge(superclass, classX)
    if len_G<len(set(G.nodes())):
        graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
        plt.show()



def find_frequent_labels(ancestors: list, G: nx.DiGraph()):
    indegrees = [G.in_degree(parent) for parent in ancestors]
    min_element = min(indegrees)
    min_indices = [i for i, x in enumerate(indegrees) if x == min_element]
    topmost_parent = [ancestors[i] for i in min_indices]
    return topmost_parent


def labels_most_fre(datas: dict):
    # frequency of each label
    label_counter = Counter(label for labels_list in datas.values() for label in labels_list)
    # Mayjor voting label
    most_common_labels = label_counter.most_common()
    # print the consequence
    # print("Label Frequencies:", most_common_labels)
    # most frequently appeared label
    max_frequency = most_common_labels[0][1]
    most_common_labels = [label for label, frequency in most_common_labels if frequency == max_frequency]
    # print("Most Common Labels:", most_common_labels)
    return most_common_labels


def ground_truth_labels(filename, top=False):
    ground_label_name1 = "groundTruth.csv"
    data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
    labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
    target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
    with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)
    row = ground_truth_csv[ground_truth_csv['fileName'] == filename].iloc[0]
    All_ancestors = []
    topmost_parent = []
    classX = row["class"]
    if row["class"] != ' ':
        if filename in labels:
            table_label = pd.read_csv(os.path.join(os.getcwd(), "datasets/TabFact/Label", filename))[
                "classLabel"].unique()
            for loa in table_label:
                ancestors = list(nx.ancestors(G, loa))
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
            All_ancestors = [classX]
        return All_ancestors


def label_dict(tables: list, is_Parent=True):
    dict_table_labels = {}
    for table in tables:
        dict_table_labels[table] = ground_truth_labels(table, top=is_Parent)
    return dict_table_labels


def no_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    if len(set1.intersection(set2)) == 0:
        return True
    else:
        return False


def simple_tree_with_cluster_label(threCluster_dict, orginal_tree, table_names):
    layer_info_dict = {}
    Parent_nodes_h = {}
    simple_tree = None
    for index, (thre, clusters) in enumerate(threCluster_dict):
        print(thre, len(clusters))
    for index, (thre, clusters) in enumerate(threCluster_dict):
        print(thre, len(clusters))
        if index == 0:
            start_time = time.time()
            parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
            end_time = time.time()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the best silhouette tree")

            # print(parent_nodes)
            simple_tree = PKL.simplify_graph(orginal_tree, parent_nodes)
            # print(simple_tree)
        start_time = time.time()
        parent_nodes = PKL.slice_tree(orginal_tree, clusters, table_names)  #
        layer_info_dict[index] = parent_nodes
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds for slice the tree")
        layer_num = len(threCluster_dict) - 1
        start_time_label = time.time()
        for cluster, (closest_parent, mutual_parent_nodes) in parent_nodes.items():
            simple_tree.nodes[closest_parent]['data'] = list(cluster)
            if index == layer_num:
                simple_tree.nodes[closest_parent]['label'] = labels_most_fre(label_dict(list(cluster), is_Parent=True))
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node parent layer'
                Parent_nodes_h[cluster] = closest_parent
            else:
                simple_tree.nodes[closest_parent]['label'] = labels_most_fre(label_dict(list(cluster), is_Parent=False))
                simple_tree.nodes[closest_parent]['type'] = 'data cluster node'

            wrong_labels = {}
            # print(simple_tree.nodes[closest_parent]['label'])
            for i in cluster:
                label_per = ground_truth_labels(i, top=True) \
                    if index == layer_num else ground_truth_labels(i, top=False)
                if no_intersection(label_per, simple_tree.nodes[closest_parent]['label']) is True:
                    wrong_labels[i] = label_per
            simple_tree.nodes[closest_parent]['Wrong_labels'] = wrong_labels
            label_cluster = simple_tree.nodes[closest_parent]['label']
            simple_tree.nodes[closest_parent]['Purity'] = "{:.3f}".format(
                (1 - len(wrong_labels.keys()) / len(cluster)) / len(label_cluster))
        end_time_label = time.time()
        elapsed_time = end_time_label - start_time_label
        print(f"Elapsed time: {elapsed_time:.4f} seconds for calculate the label of the tree")
    return simple_tree, layer_info_dict, Parent_nodes_h


def print_path_label(tree, layer_info, Parent_nodes):
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
                                    if len(ancestors)==0:
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
    tree_consistency_score = "{:.3f}".format(sum(all_pathCompatibility) / len(all_pathCompatibility))
    print("the metric is ", tree_consistency_score, "all_path_number is ", len(all_pathCompatibility))

    return tree_consistency_score


def tree_consistency_metric(embedding_file, dataset):
    EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", dataset)
    with open(os.path.join(EMBEDDING_FOLDER, embedding_file), 'rb') as file:
        data = pickle.load(file)
    data_path = os.path.join(os.getcwd(), "datasets", dataset, "Try.csv")
    ground_truth_csv = pd.read_csv(data_path, encoding='latin1')
    table_with_available_labels = ground_truth_csv.iloc[:, 0].unique()
    data = [i for i in data if i[0] in table_with_available_labels][0:1500]
   
    table_names = [i[0] for i in data if i[0] in table_with_available_labels][0:1500]
    
    encodings = np.array([np.mean(i[1], axis=0) for i in data])

    linkage_matrix = sch.linkage(encodings, method='complete', metric='euclidean')

    folder = "fig/" + dataset
    result_folder = os.path.join("result/Valerie", dataset)

    dendrogra = PKL.plot_tree(linkage_matrix, folder, node_labels=table_names)
    tree_test = PKL.dendrogram_To_DirectedGraph(encodings, linkage_matrix, table_names)
    start_time = time.time()
    threCluster_dict = PKL.best_clusters(dendrogra, linkage_matrix, encodings,estimate_num_cluster=300)



    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")
    simple_tree, layer_info_dict, Parent_nodes_h = \
        simple_tree_with_cluster_label(threCluster_dict, tree_test, table_names)
    start_time = time.time()
    TCS = print_path_label(simple_tree, layer_info_dict[0], Parent_nodes_h)
    end_time = time.time()
    print(f"Embedding file {embedding_file}", f", Tree Consistency metric: {TCS}")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds for finding the best clusters")
    PKL.print_clusters_top_down(simple_tree, layer_info_dict)

    Purity_layers = PKL.purity_per_layer(simple_tree, layer_info_dict)
    for layer, purity_list in Purity_layers.items():
        purity_list = [float(i) for i in purity_list]
        Purity_layer = "{:.3f}".format(sum(purity_list) / len(purity_list))
        print(f"layer {layer} purity is: {Purity_layer}")
    file_path = os.path.join(result_folder, embedding_file[0:-4] + "_results.pkl")

    with open(file_path, 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump((dendrogra, linkage_matrix, threCluster_dict,
                     simple_tree, tree_test, layer_info_dict, Parent_nodes_h), file)

"""
labels_dict = {
    's1': ['a', 'b', 'c', 'd'],
    's2': ['a', 'b', 'd', 'f'],
    's3': ['a', 'b', 'f', 'e', 'g'],
    's4': ['a', 'b', 'h']
}

most_common_labels = labels_most_fre(labels_dict)
print(most_common_labels)

table_e = ['2-14080161-3.html.csv', '2-16900662-5.html.csv', '2-18938222-1.html.csv',
           '2-154957-5.html.csv', '2-18652198-10.html.csv', '2-18938222-5.html.csv', '2-13535631-7.html.csv']

most_common_labels = labels_most_fre(label_dict(table_e, is_Parent=False))
print(most_common_labels)
dataset = "TabFact"
EMBEDDING_FOLDER = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", dataset)
for embedding in [i for i in os.listdir(EMBEDDING_FOLDER) if i.endswith("pkl") and
                                                             "sbert" in i][0:]:
    tree_consistency_metric(embedding, dataset)
    break"""
