import math
import os
import pickle
from argparse import Namespace
import networkx as nx
import numpy as np
import pandas as pd

from EndToEnd.T import generateUML
from EndToEnd.UML import savefig_uml
#from interactiveFigure import draw_interactive_graph
from clustering import most_frequent_list, clustering, data_classes
from ClusterHierarchy.ClusterDecompose import tree_consistency_metric
from TableCluster.tableClustering import typeInference
from Utils import findSubCol, name_type, mkdir, calculate_similarity
from ClusterHierarchy.JaccardMetric import JaccardMatrix


def find_node_tables(G, N):
    children_with_type_data = []
    descendants = set(nx.dfs_preorder_nodes(G, source=N)) - {N}
    for descendant in descendants:
        if G.nodes[descendant].get('type') == 'data':
            children_with_type_data.append(descendant)
    return children_with_type_data


def checkTree(tree: nx.DiGraph()):
    #draw_interactive_graph(tree)
    for node in tree.nodes():
        if tree.nodes[node].get("type") != 'data':
            tables = find_node_tables(tree,node )
            print(tables)
            print(node, tree.nodes[node])


def find_attribute(node_tables, colDict, SubcolDict, filePath: str):
    def findCluster(column, columnDict):
        index_attri = -1
        for key, cluster_columns in columnDict.items():
            if column in cluster_columns:
                index_attri = key
                break
        return index_attri

    if len(node_tables) == 0:
        return [], []
    attribute_index_dict = {}
    for table in node_tables:
        subject_col = findSubCol(SubcolDict, table + ".csv")
        columns = pd.read_csv(os.path.join(filePath, table + ".csv")).columns
        if subject_col is not None:
            # subject_columns.append(subject_col)
            columns = [i for i in columns if i != subject_col]
        for col in columns:
            attribute_index = findCluster(f"{table}.{col}", colDict)
            if attribute_index not in attribute_index_dict:
                attribute_index_dict[attribute_index] = 1
            else:
                attribute_index_dict[attribute_index] += 1
    limit_attri = 0###TODO maybe need to be soft coded later  len(node_tables) / 2
    keep_keys = [key for key in attribute_index_dict.keys() if attribute_index_dict[key] >= limit_attri]
    return keep_keys


def update_attributes(tree: nx.DiGraph(), colDict, SubcolDict, filePath, name_dict):
    for node in tree.nodes():
        tables = find_node_tables(tree, node)
        co_index = find_attribute(tables, colDict, SubcolDict, filePath)
        tree.nodes[node]['attributes'] = co_index
        names_node = name_type(tables, name_dict)
        tree.nodes[node]['name'] = names_node


def hierarchy(cluster_dict, hp: Namespace, name_dict, limit=39):
    F = open(f"datasets/{hp.dataset}/SubjectCol.pickle", 'rb')
    SE = pickle.load(F)
    datafile_path = f"result/embedding/{hp.dataset}/"
    filepath = f"datasets/{hp.dataset}/Test/"
    F = open(datafile_path + hp.P23Embed, 'rb')
    content = pickle.load(F)
    store_path = f"/result/EndToEnd/{hp.dataset}/"
    mkdir(store_path)
    for name in cluster_dict.keys():
        cluster_info = cluster_dict[name]
        cluster = cluster_info["cluster"]
        names = []
        input_data = []
        for table_name in cluster:
            column_table = pd.read_csv(os.path.join(filepath, table_name + ".csv")).columns
            subcol = findSubCol(SE, table_name + ".csv")
            column_table_combine = [f"{table_name}.{i}" for i in column_table if i != subcol]
            # names.extend(column_table)
            names.extend(column_table_combine)
            if hp.P23Embed.endswith("_column.pkl"):
                input_data.extend([i[1][0] for i in content if i[0] in column_table_combine])
        MIN =  math.ceil(len(input_data) / 40) if  math.ceil(len(input_data) / 40)  >2 else 2
        colcluster_dict = clustering(input_data, names,MIN,"Agglomerative",
                                     max=2 *MIN )
        print("attribute clusters ", len(colcluster_dict))
        colCluster = {index: {'name': name_type([i.split(".")[1] for i in cluster]), 'cluster': cluster} for
                      index, cluster in colcluster_dict.items()}
        data_path = os.getcwd() + "/datasets/%s/Test/" % hp.dataset
        jaccard_score = JaccardMatrix(colcluster_dict, data_path)[2]
        simple_tree = None
        if len(cluster) > limit:
            print("\n", name, len(cluster_info["cluster"]), cluster_info["TopLabel"], cluster_info["TPurity"],
                  cluster_info["LowLabel"], cluster_info["name"][0])
            TCS, ALL_path, simple_tree = tree_consistency_metric(cluster, jaccard_score, hp.P23Embed,
                                                                 hp.dataset, sliceInterval=hp.intervalSlice,
                                                                 delta=hp.delta,
                                                                 endtoEnd=True)
            update_attributes(simple_tree, colcluster_dict, SE, filepath, name_dict)
        cluster_dict[name]["attributes"] = colCluster
        cluster_dict[name]["tree"] = simple_tree
        #if simple_tree is not None:
           # checkTree(simple_tree)
        # break


def read_col_Embeddings(embedding_file, dataset, selected_tables=None):
    print("embedding_file", embedding_file)
    data_path = os.path.join(f"datasets/{dataset}", "Test")
    datafile_path = os.getcwd() + "/result/embedding/" + dataset + "/"
    if selected_tables is None:
        selected_tables = [i for i in os.listdir(data_path) if i.endswith(".csv")]
    F = open(os.path.join(datafile_path, embedding_file), 'rb')
    if embedding_file.endswith("_column.pkl"):
        content = pickle.load(F)
        content = {i[0]: i[1][0] for i in content}
    else:
        original_content = pickle.load(F)
        content_dict = {i[0]: i[1] for i in original_content}
        content = {}
        for fileName in selected_tables:
            table_embedding = content_dict[fileName]
            table = pd.read_csv(os.path.join(data_path, fileName))
            for index, col in enumerate(table.columns):
                content[f"{fileName[:-4]}.{col}"] = table_embedding[0]
    return content


def endToEndRelationship(hp: Namespace, Types_clusters: dict):
    """
    This is the end-to-end relationship discovery between attributes
    :param Types_clusters: format: {index:{'cluster': [T1,T2,T3], 'name': TypeName,
    'TopLabel': GroundTruthLabel, 'LowLabel':Lowest ground truth label, 'TPurity': Purity,
    'attributes': column attribute clusters, 'tree': the decomposed hierarchy inside the type
    }, ...}
    dataset: selected dataset
    embedding_file: the embedding pickle file

    Returns:
        relationship pairs
    """
    embedding_file = hp.P4Embed

    def cluster_relationship(subject_attributes, object_attributes_clusters, threshold1, threshold2, isEuclidean=False):
        result_index = []
        for index, object_attributes_cluster in object_attributes_clusters.items():
            portion_object = []
            for subject in subject_attributes:
                for index_object, object_attribute in enumerate(object_attributes_cluster):
                    if calculate_similarity(subject, object_attribute, Euclidean=isEuclidean) > threshold1:
                        if index_object not in portion_object:
                            portion_object.append(index_object)
            if len(portion_object) / len(object_attributes_cluster) > threshold2:
                result_index.append(index)
        return result_index

    def find_cluster_embeddings(contentEmbedding, attribute_clusters):
        embeddings = {}
        for index_attri, attri_info in attribute_clusters.items():
            attribute_cluster = attri_info['cluster']
            embeddings[index_attri] = np.array([contentEmbedding[j] for j in attribute_cluster])
        return embeddings

    def read_types_embeddings(clusterInfo, content_embedding, ind):
        subCols = clusterInfo[ind]['subjectAttribute']['attributes']
        subcols_embedding = np.array([content_embedding[sub] for sub in subCols])
        attribute_clusters = clusterInfo[ind]['attributes']
        attribute_clusters_embedding = find_cluster_embeddings(content_embedding, attribute_clusters)
        return subcols_embedding, attribute_clusters_embedding

    content = read_col_Embeddings(embedding_file, hp.dataset)
    keys = list(Types_clusters.keys())
    for key in keys:
        Types_clusters[key]["relationship"] = {}

    for index, index_i in enumerate(keys):
        subcolI_embedding, attribute_clusters_embeddingI = read_types_embeddings(Types_clusters, content, index_i)
        for index_j in keys[index + 1:]:
            subcolJ_embedding, attribute_clusters_embeddingJ = read_types_embeddings(Types_clusters, content, index_j)
            relationship_indexI = cluster_relationship(subcolI_embedding, attribute_clusters_embeddingJ, hp.similarity,
                                                       hp.portion)
            relationship_indexJ = cluster_relationship(subcolJ_embedding, attribute_clusters_embeddingI, hp.similarity,
                                                       hp.portion)
            if relationship_indexI:
                Types_clusters[index_i]["relationship"][index_j] = relationship_indexI
                print(f"Type{index_i} to Type {index_j}'s relationship lies in attributes index {relationship_indexI}")
            if relationship_indexJ:
                Types_clusters[index_j]["relationship"][index_i] = relationship_indexJ
                print(f"Type{index_j} to Type {index_i}'s relationship lies in attributes index {relationship_indexJ}")


def check_model_num(Type_dict):
    total_num = 0
    for index, type_dict in Type_dict.items():
        tree = type_dict['tree']
        if tree is not None:
            subtypes = [i for i in tree.nodes() if
                        tree.nodes[i].get('type') != 'data']
            print(f"subtypes in side this type {index} are {len(subtypes)}")
            total_num += len(subtypes)
        else:
            total_num += 1
    return total_num


def endToEnd(hp: Namespace):
    def name_types(cluster_dict, name_dict=None):
        new_cluster_dict = {}
        for i, cluster in cluster_dict.items():
            name_i = name_type(cluster, name_dict)
            new_cluster_dict[i] = {'cluster': cluster, 'name': name_i}
        return new_cluster_dict

    def type_info(typeDict, nameDict, subcol_dict):
        gt_clusters = data_classes(f"datasets/{hp.dataset}/Test", f"datasets/{hp.dataset}/groundTruth.csv")[0]
        gt_clusters_low = \
            data_classes(f"datasets/{hp.dataset}/Test", f"datasets/{hp.dataset}/groundTruth.csv", superclass=False)[0]
        new_cluster_dict = name_types(typeDict, nameDict)
        for index in new_cluster_dict.keys():
            info_dict = new_cluster_dict[index]
            tables = new_cluster_dict[index]['cluster']
            subCols = [f"{table_name}.{findSubCol(subcol_dict, table_name + '.csv')}" for table_name in tables if
                       findSubCol(subcol_dict, table_name + '.csv') is not None]
            subcol_name = name_type([i.split(".")[1] for i in subCols])
            new_cluster_dict[index]["subjectAttribute"] = {'name': subcol_name, 'attributes': subCols}
            gt_labels = most_frequent_list([gt_clusters[i] for i in info_dict["cluster"]])
            gt_labels_low = most_frequent_list([[gt_clusters_low[i]] for i in info_dict["cluster"]])
            info_dict["TopLabel"] = gt_labels
            info_dict["TPurity"] = len(
                [i for i in info_dict["cluster"] if bool(set(gt_clusters[i]).intersection(set(gt_labels)))])
            info_dict["LowLabel"] = gt_labels_low
        return new_cluster_dict

    # TODO hard coded part needs to change later
    datafile_path = os.getcwd() + "/result/embedding/" + hp.dataset + "/"
    dict_file = typeInference(hp.P1Embed, hp, datafile_path)
    cluster_dict = dict_file[hp.clustering]
    name_dict = {row["table"]: row["name"] for index, row in
                 pd.read_csv(f"datasets/{hp.dataset}/naming.csv").iterrows()}
    F = open(f"datasets/{hp.dataset}/SubjectCol.pickle", 'rb')
    SE = pickle.load(F)
    cluster_dict_all = type_info(cluster_dict, name_dict, SE)
    print("top level type number: ", len(cluster_dict),len(cluster_dict_all))
    del cluster_dict
    hierarchy(cluster_dict_all, hp, name_dict)
    print("endtoEnd ...")
    endToEndRelationship(hp, cluster_dict_all)
    number = check_model_num(cluster_dict_all)
    print("infer model number", number)
    path = os.path.join(os.getcwd(),f"result\WDCEndtoEnd.xlsx")
    print(path)
    with open(path, 'wb') as f:
          pickle.dump(cluster_dict_all, f)
