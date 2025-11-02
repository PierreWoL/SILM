import math
import os
import pickle
from argparse import Namespace
import networkx as nx
import numpy as np
import pandas as pd

from EndToEnd.T import generateUML
from EndToEnd.UML import savefig_uml
from Naming import GPTnaming
#from interactiveFigure import draw_interactive_graph
from clustering import most_frequent_list, clustering, data_classes
from ClusterHierarchy.ClusterDecompose import tree_consistency_metric
from TableCluster.tableClustering import typeInference
from Utils import findSubCol, name_type, mkdir, calculate_similarity
from ClusterHierarchy.JaccardMetric import JaccardMatrix


def read_col_Embeddings(embedding_file, dataset, selected_tables=None):
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
                content[f"{fileName}.{col}"] = table_embedding[
                    0]  # content[f"{fileName[:-4]}.{col}"] = table_embedding[0]
    return content


def find_node_tables(G, N):
    children_with_type_data = []
    descendants = set(nx.dfs_preorder_nodes(G, source=N)) - {N}
    for descendant in descendants:
        if G.nodes[descendant].get('type') == 'data':
            children_with_type_data.append(descendant)
    return children_with_type_data


def checkTree(tree: nx.DiGraph(), colDict,name=None):
    """if name!=None:
        draw_interactive_graph(tree,file_path=f"result/EndToEnd/WDC/{name}.html")
    else:
        draw_interactive_graph(tree)"""
    for node in tree.nodes():
        if tree.nodes[node].get("type") != 'data':
            tables = find_node_tables(tree, node)
            attribute_dict = tree.nodes[node]["attributes_dict"]
            name = tree.nodes[node]["name"]
            label = tree.nodes[node]["label"]
            Purity = tree.nodes[node]["Purity"]
            print("\n", node, " ", name, f"{label} Purity {Purity} ({len(tables)})")
            for key, number in attribute_dict.items():
                names = colDict[key]['name']
                attri_name = names[0] if len(names) > 0 else 'unnamed'
                print(f"{attri_name} ({number})")


def find_attribute(node_tables, colDict, filePath: str):
    def findCluster(column, columnDict):
        index_attri = -1
        for key, cluster_columns in columnDict.items():
            if column in cluster_columns:
                index_attri = key
                break
        return index_attri

    if len(node_tables) == 0:
        return [], {}
    attribute_index_dict = {}
    for table in node_tables:
        subject_cols = findSubCol(filePath, table + ".csv")
        columns = pd.read_csv(os.path.join(filePath, table + ".csv")).columns
        if subject_cols is not []:
            columns = [i for i in columns if i not in subject_cols]
        for col in columns:
            attribute_index = findCluster(f"{table}.{col}", colDict)
            if attribute_index not in attribute_index_dict:
                attribute_index_dict[attribute_index] = 1
            else:
                attribute_index_dict[attribute_index] += 1
    limit_attri = 0  ###TODO maybe need to be soft coded later  len(node_tables) / 2
    keep_keys = [key for key in attribute_index_dict.keys() if attribute_index_dict[key] >= limit_attri]
    return keep_keys, attribute_index_dict


def update_attributes(tree: nx.DiGraph(), colDict, filePath, name_dict):
    for node in tree.nodes():
        tables = find_node_tables(tree, node)
        co_index, attri_keys = find_attribute(tables, colDict, filePath)
        tree.nodes[node]['attributes'] = co_index
        tree.nodes[node]['attributes_dict'] = {key: value for key, value in attri_keys.items() if key in co_index}
        names_node = name_type(tables, name_dict)
        tree.nodes[node]['name'] = names_node


def find_cluster_embeddings(cluster, content, filepath: str):
    names = []
    input_data = []
    for table_name in cluster:
        if table_name+".csv" in os.listdir(os.path.join(filepath)):
            column_table = pd.read_csv(os.path.join(filepath, table_name + ".csv")).columns
            subcols = findSubCol(filepath, table_name + ".csv")
            column_table_combine = [f"{table_name}.{i}" for i in column_table if i not in subcols]
            # names.extend(column_table)
            names.extend(column_table_combine)
            input_data.extend([embed for key, embed in content.items() if key in column_table_combine])
    return input_data, names



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
            embeddings[index_attri] = np.array([contentEmbedding[j] for j in attribute_cluster if j in contentEmbedding])
        return embeddings

    def read_types_embeddings(clusterInfo, content_embedding, ind):
        subCols = clusterInfo[ind]['subjectAttribute']['attributes']
        subcols_embedding = np.array([content_embedding[sub] for sub in subCols if sub in content_embedding])
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
                name_i = Types_clusters[index_i]['name']
                print(f"Type{index_i} to Type {index_j}'s relationship lies in attributes index {relationship_indexI}")
            if relationship_indexJ:
                Types_clusters[index_j]["relationship"][index_i] = relationship_indexJ
                name_j = Types_clusters[index_j]['name']
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


def name_types(cluster_dict, name_dict=None):
    new_cluster_dict = {}
    for i, cluster in cluster_dict.items():
        name_i = name_type(cluster, name_dict)
        new_cluster_dict[i] = {'cluster': cluster, 'name': name_i}
    return new_cluster_dict


def type_info(typeDict, dataset, nameDict=None, noLabel=False):
    gt_clusters = data_classes(f"datasets/{dataset}/Test", f"datasets/{dataset}/groundTruth.csv")[0]

    gt_clusters_low = \
        data_classes(f"datasets/{dataset}/Test", f"datasets/{dataset}/groundTruth.csv", superclass=False)[0]
    new_cluster_dict = name_types(typeDict, nameDict) if nameDict is not None \
        else {i: {'cluster': cluster, 'name': None} for i, cluster in typeDict.items()}
    for index in new_cluster_dict.keys():
        info_dict = new_cluster_dict[index]
        tables = new_cluster_dict[index]['cluster']
        subCols = []
        for table_name in tables:
            per_sub_cols = findSubCol(f"datasets/{dataset}/Test/", table_name + '.csv')
            if per_sub_cols is not []:
                subCols.extend([f"{table_name}.{per_sub_col}" for per_sub_col in per_sub_cols])
        subcol_name = name_type([i.split(".")[1] for i in subCols])
        new_cluster_dict[index]["subjectAttribute"] = {'name': subcol_name, 'attributes': subCols}

        if noLabel is False:
            gt_labels = most_frequent_list([gt_clusters[i] for i in info_dict["cluster"]])
            gt_labels_low = most_frequent_list([[gt_clusters_low[i]] for i in info_dict["cluster"]])

            info_dict["TopLabel"] = gt_labels
            info_dict["LowLabel"] = gt_labels_low
            info_dict["TPurity"] = len(
                [i for i in info_dict["cluster"] if bool(set(gt_clusters[i]).intersection(set(gt_labels)))])
            info_dict["LPurity"] = len(
                [i for i in info_dict["cluster"] if bool({gt_clusters_low[i]}.intersection(set(gt_labels_low)))])

        new_cluster_dict[index] = info_dict
    return new_cluster_dict


def endToEnd(hp: Namespace):
    # TODO hard coded part needs to change later
    dict_file = typeInference(hp.P1Embed, hp)
    cluster_dict = dict_file[hp.clustering]
    name_dict = {row["table"]: row["name"] for index, row in
                 pd.read_csv(f"datasets/{hp.dataset}/naming.csv").iterrows()}
    cluster_dict_all = type_info(cluster_dict, hp.dataset, nameDict=name_dict)

    print("top level type number: ", len(cluster_dict), len(cluster_dict_all))
    del cluster_dict
    hierarchy(cluster_dict_all, hp, name_dict)
    print("endtoEnd ...")
    endToEndRelationship(hp, cluster_dict_all)
    number = check_model_num(cluster_dict_all)
    print("infer model number", number)
    path = os.path.join(os.getcwd(), f"result\WDCEndtoEnd.pkl")
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(cluster_dict_all, f)
    #uml_code,sub_umls = generateUML(cluster_dict_all, hp.dataset, hp.estimateNumber)
    #mkdir(f"result/EndToEnd/{hp.dataset}/{hp.estimateNumber}")
    #savefig_uml(uml_code, os.getcwd(), fileName=f"result/EndToEnd/{hp.dataset}/{hp.estimateNumber}/all.uml")


def hierarchy(cluster_dict, hp: Namespace, name_dict, limit=32):
    filepath = f"datasets/{hp.dataset}/Test/"
    content = read_col_Embeddings(hp.P23Embed, hp.dataset)
    store_path = f"/result/EndToEnd/{hp.dataset}/"
    mkdir(store_path)
    topType_dict = {}
    for name in cluster_dict.keys():
        cluster_info = cluster_dict[name]
        cluster = cluster_info["cluster"]
        input_data, names = find_cluster_embeddings(cluster, content, filepath)
        MIN = math.ceil(len(input_data) / 40) if math.ceil(len(input_data) / 40) > 2 else 2
        colcluster_dict = clustering(input_data, names, MIN, hp.clustering,
                                     max=2 * MIN + 5)
        colcluster_dict = dict(sorted(colcluster_dict.items(), key=lambda item: len(item[1]), reverse=True))
        SA_name_attri = cluster_dict[name]["subjectAttribute"]["name"][0] if len(
            cluster_dict[name]["subjectAttribute"]["name"]) > 0 else "? Name"
        gt_clusters_low = \
            data_classes(f"datasets/WDC/Test", f"datasets/WDC/groundTruth.csv", superclass=False)[0]
        tables = {i: gt_clusters_low[i] for i in cluster_dict[name]["cluster"]}
        topType_dict[name] = {'name':cluster_dict[name]["name"][0], 'Size':len(cluster_dict[name]["cluster"]), 'TopLabel':cluster_dict[name]["TopLabel"],
                              'TPurity':cluster_dict[name]["TPurity"] / len(cluster_dict[name]["cluster"]),
                              'LowLabel':cluster_dict[name]["LowLabel"],
                              'LPurity':cluster_dict[name]["LPurity"] / len(cluster_dict[name]["cluster"])}

        colCluster = {index: {'name': name_type([i.split(".")[1] for i in cluster]), 'cluster': cluster} for
                      index, cluster in colcluster_dict.items()}


        data_path = os.getcwd() + "/datasets/%s/Test/" % hp.dataset
        jaccard_score = JaccardMatrix(colcluster_dict, data_path)[2]
        simple_tree = None
        if len(cluster) > limit:
            print(f'hierarchy'
                  f'\n{cluster_dict[name]["name"][0]} \n {cluster_dict[name]["TopLabel"]}  '
                  f'{cluster_dict[name]["TPurity"] / len(cluster_dict[name]["cluster"])} '
                  f'{cluster_dict[name]["LowLabel"]} {cluster_dict[name]["LPurity"] / len(cluster_dict[name]["cluster"])} \n '  # {tables} \n
                  f'{SA_name_attri} ({len(cluster_dict[name]["subjectAttribute"]["attributes"])})')
            for index, colcluster_info in colCluster.items():
                name_attri = colcluster_info['name'][0] if len(colcluster_info['name']) > 0 else "Unnamed attribute"
                print(f"{name_attri} ({len(colcluster_info['cluster'])})")


            TCS, ALL_path, simple_tree = tree_consistency_metric(cluster, jaccard_score, hp.P23Embed,
                                                                 hp.dataset, sliceInterval=hp.intervalSlice,
                                                                 delta=hp.delta,
                                                                 store_results=False)
            if simple_tree is not None:
                update_attributes(simple_tree, colcluster_dict,  filepath, name_dict)
        cluster_dict[name]["attributes"] = colCluster
        cluster_dict[name]["tree"] = simple_tree
        if simple_tree is not None:
            checkTree(simple_tree, colCluster)
    df = pd.DataFrame.from_dict(topType_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)

    df.to_csv(os.getcwd() + "/result/EndToEnd/TopTypes.csv", index=False)

        # break

def hierarchy_new(cluster_dict, hp: Namespace, limit=40):
    filepath = f"datasets/{hp.dataset}/Test/"
    content = read_col_Embeddings(hp.P23Embed, hp.dataset)
    store_path = f"result/EndToEnd/{hp.dataset}/"
    mkdir(store_path)
    topType_dict = {}
    for name in cluster_dict.keys():
        print("current cluster is ,", name,  cluster_dict[name]["name"])
        cluster_info = cluster_dict[name]
        cluster = cluster_info["cluster"]
        input_data, names = find_cluster_embeddings(cluster, content, filepath)
        MIN = math.ceil(len(input_data) / 40) if math.ceil(len(input_data) / 40) > 2 else 2
        colcluster_dict = clustering(input_data, names, MIN, hp.clustering,
                                     max=2 * MIN + 5)
        # print("attribute clusters ", len(colcluster_dict))
        colcluster_dict = dict(sorted(colcluster_dict.items(), key=lambda item: len(item[1]), reverse=True))
        SA_name_attri = cluster_dict[name]["subjectAttribute"]["name"][0] if len(
            cluster_dict[name]["subjectAttribute"]["name"]) > 0 else "? Name"
        gt_clusters_low = \
            data_classes(f"datasets/{hp.dataset}/Test", f"datasets/{hp.dataset}/groundTruth.csv", superclass=False)[0]
        tables = {i: gt_clusters_low[i] for i in cluster_dict[name]["cluster"]}
        # print("cluster_dict", cluster_dict)
        topType_dict[name] = {'Size':len(cluster_dict[name]["cluster"])}
        if isinstance(cluster_dict[name]["name"], str):
             topType_dict[name]['name']= cluster_dict[name]["name"]
        else:
            topType_dict[name]['name'] = cluster_dict[name]["name"][0]
        colCluster = {index: {'name': name_type([i.split(".")[1] for i in cluster]), 'cluster': cluster} for
                      index, cluster in colcluster_dict.items()}
        #print(cluster_info["name"],len(cluster), )
        #for key, info in colCluster.items():
            #print(info['name'])
        data_path = os.getcwd() + "/datasets/%s/Test/" % hp.dataset
        jaccard_score = JaccardMatrix(colcluster_dict, data_path)[2]
        simple_tree = None
        for index, colcluster_info in colCluster.items():
            name_attri = colcluster_info['name'][0] if len(colcluster_info['name']) > 0 else "Unnamed attribute"
        if len(cluster) > limit:
            for index, colcluster_info in colCluster.items():
                name_attri = colcluster_info['name'][0] if len(colcluster_info['name']) > 0 else "Unnamed attribute"
                #print(f"{name_attri} ({len(colcluster_info['cluster'])})")


            TCS, ALL_path, simple_tree = tree_consistency_metric(cluster, jaccard_score, hp.P23Embed,
                                                                 hp.dataset, sliceInterval=hp.intervalSlice,
                                                                 delta=hp.delta,
                                                                 store_results=False)
            if simple_tree is not None:
                    for node in simple_tree.nodes():
                        tables = find_node_tables(simple_tree, node)
                        cluster_df = read_clusters_tables(hp.dataset, tables)
                        co_index, attri_keys = find_attribute(tables, colcluster_dict, filepath)
                        simple_tree.nodes[node]['attributes'] = co_index
                        simple_tree.nodes[node]['attributes_dict'] = {key: value for key, value in attri_keys.items() if
                                                               key in co_index}
                        names_node = gptname(cluster_df, format=3, sampling=None, sampling_number=5,
                                             header=True, table_names=names, instruction=instruction, shorts=None,
                                             AI_instruction=AI_Ins)
                        simple_tree.nodes[node]['name'] = names_node

        cluster_dict[name]["attributes"] = colCluster
        cluster_dict[name]["tree"] = simple_tree
        if simple_tree is not None:
            name_cluster = cluster_info["name"]
            #checkTree(simple_tree, colCluster,name=f"{name}_{name_cluster}")
    print("topType_dict",topType_dict)
    df = pd.DataFrame.from_dict(topType_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    df.to_csv("E:/Project/CurrentDataset/result/EndToEnd/GDS/endToend.csv")

def gptname(cluster_df,format =1, sampling=None,sampling_number=5,  header=True, table_names=None, instruction = None, shorts = None, AI_instruction=None):
        task_element = ""
        if format == 2:
            task_element = "tables"
        elif format == 3:
            task_element = "tables with <sc> and </sc> marking the subject attributes"
        elif format == 4:
            task_element = "subject attributes of tables"
        naming = GPTnaming(apiKey="", format=format,
                           sampling=sampling, sampling_number=sampling_number, header=header, table_names=table_names)
        task = (
            f"Given a cluster of {task_element} separated by <table> ... </table>, identify and provide a name for the conceptual type that best describes the tables in the cluster.")
        reply = naming.generate_answers(cluster_df, task, reply=None, instructions=instruction, shorts=shorts,
                                        AI_instructions=AI_instruction, newMessage=True)
        return reply
def read_clusters_tables(dataset, cluster):
    tables = []
    for table_name in cluster:
        table = pd.read_csv(f"datasets/{dataset}/Test/{table_name}.csv")
        tables.append(table)
    return tables
def ttt(hp: Namespace):
    cluster_dict = typeInference(hp.P1Embed, hp)["Agglomerative"]
    # print(cluster_dict)
    results = pd.read_csv(f"result/P1/{hp.dataset}/All/{hp.P1Embed[:-4]}/purityCluster1.csv")
    naming5_dict = {}
    for index, row in results.iterrows():
        naming5_value = row['Naming4']
        if naming5_value not in naming5_dict:
            naming5_dict[naming5_value] = []
        cluster_id = row['Cluster id']
        naming5_dict[naming5_value].append(cluster_id)

    new_clusters = {}
    for name, ids in naming5_dict.items():
        if len(ids) ==1:
            new_clusters[ids[0]] = {}
            new_clusters[ids[0]]["cluster"] = cluster_dict[ids[0]]
        else:
            id_new = ids[0]
            new_clusters[id_new] = {}
            new_clusters[ids[0]]["cluster"] = cluster_dict[id_new]
            for id in ids[1:]:
                new_clusters[id_new]["cluster"].extend(cluster_dict[id])
    for id in new_clusters.keys():
        cluster_info = new_clusters[id]
        names = [name_dict[i] for i in cluster_info["cluster"]]
        cluster_df = read_clusters_tables(hp.dataset,cluster_info["cluster"])
        reply = gptname(cluster_df, format=3,sampling=None,sampling_number=5,
                        header=True, table_names=names, instruction = instruction, shorts = None, AI_instruction=AI_Ins)
        new_clusters[id]["name"]=reply

        subCols = []
        for table_name in cluster_info["cluster"]:
            per_sub_cols = findSubCol(f"datasets/{hp.dataset}/Test/", table_name+".csv")
            if per_sub_cols is not []:
                subCols.extend([f"{table_name}.{per_sub_col}" for per_sub_col in per_sub_cols])
        subcol_name = name_type([i.split(".")[1] for i in subCols])
        new_clusters[id]["subjectAttribute"] = {'name': subcol_name, 'attributes': subCols}
        print("iterate the cluster",id,reply, len(cluster_df))
    hierarchy_new(new_clusters, hp)
    endToEndRelationship(hp, new_clusters)
    with open(f"result/P1/{hp.dataset}/All/{hp.P1Embed[:-4]}/merged.pkl", "wb") as f:
        pickle.dump(new_clusters, f)
    uml_code, sub_umls = generateUML(new_clusters, hp.dataset, hp.estimateNumber)
    mkdir(f"result/EndToEnd/{hp.dataset}/{hp.estimateNumber}")
    # savefig_uml(uml_code, os.getcwd(), fileName=f"result/EndToEnd/{hp.dataset}/{hp.estimateNumber}/allSwAV.uml")

"""

"""
dataset="GDS"
with open(f"datasets/{dataset}/graphGroundTruth.pkl", "rb") as f:
    graph = pickle.load(f)
top_level_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

AI_Ins = [
        f"The tables are web tables. The cluster possibly indicates an top level conceptual type in Schema.org. Schema.org's top-level types are {top_level_nodes}"]  # their closest children types
instruction = [
        "Background: The cluster of tables is derived from a hierarchical clustering algorithm applied to the embeddings of tables."
        "This clustering indicates a mutual conceptual entity type among the tables.",
        "Data Provided: all tables in the cluster."
        "Each table is separated by <table> </table> ."
        "Each TABLE follows the format: <table> TABLE_INDEX: HEADER1|HEADER2|HEADER3|\n Instance11| Instance12|Instance13| \n Instance21| Instance22|Instance23| \n....</table>.",

        "Task: Analyze the topic/theme of each of the provided table within the cluster."
        "Based on the context and theme of the tables, determine the mutual entity type of the tables. Provide a suitable name that describes this mutual entity type.",

        "Expected Output: The output should be a single word or a short phrase that clearly describes the mutual "
        "entity type of the tables in the cluster."
        "The name should be descriptive and relevant to the context of the data. AND NO OTHER TEXT!"]

names_all = pd.read_csv(f"datasets/{dataset}/naming.csv")
name_dict = {str(iter_n[0])[:-4]: iter_n[1] for index, iter_n in names_all.iterrows()}
# print(name_dict)

