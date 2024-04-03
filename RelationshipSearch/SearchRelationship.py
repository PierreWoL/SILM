import sys
from concurrent.futures import ThreadPoolExecutor
import pickle
from argparse import Namespace
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from readPKL import findBestSilhouetteDendro, sliced_clusters
from SubjectColDetect import subjectColumns
from Utils import mkdir, most_frequent
from TableCluster.tableClustering import column_gts
from RelationshipSearch.SimilaritySearch import group_files, entityTypeRelationship
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from Utils import most_frequent


def group_files_by_superclass(df):
    superclass_dict = {}
    for superclass in df['superclass'].unique():
        files = df[df['superclass'].apply(lambda x: x == superclass)]['fileName'].tolist()
        superclass_dict[str(superclass)] = files
    return superclass_dict


def find_labels(col_list, gtclusters):
    labels = []
    for column in col_list:
        if column in gtclusters.keys():
            label = gtclusters[column]
            if type(label) is list:
                for item_label in label:
                    labels.append(item_label)
            else:
                labels.append(label)
    if len(labels) == 0:
        cluster_label = None
    else:
        cluster_label = most_frequent(labels)
    return cluster_label


def find_conceptualAttri(cols_dict, table, attributes, gtclusters=None):
    """
    cols_dict: {'EventName': ['SOTAB_0.0', 'SOTAB_111.0', 'SOTAB_119.0',], 'organizer': []...}
    or {0: ['SOTAB_0.0', 'SOTAB_111.0', 'SOTAB_119.0',], 1: []...}
    table: table name
    attributes: attributes of table in the similarity dict

    """
    combines = [f"{table[:-4]}.{attribute}" for attribute in attributes]
    if gtclusters is None:
        conceptualAttributes = [cols_dict[i] for i in combines if i in cols_dict.keys()]
    else:
        cluster = [key for i in combines for key, value in cols_dict.items() if i in value]
        conceptualAttributes = [find_labels(cols_dict[i], gtclusters) for i in cluster]
    return conceptualAttributes


def calculate_linkage_distance(cluster_a, cluster_b, method='single'):
    if method == 'single':
        # 最近邻距离
        return np.min(cdist(cluster_a, cluster_b, metric='euclidean'))
    elif method == 'average':
        # 平均链接距离
        return np.mean(cdist(cluster_a, cluster_b, metric='euclidean'))
    elif method == 'complete':
        # 最远邻距离
        return np.max(cdist(cluster_a, cluster_b, metric='euclidean'))


def labels_found(clu_list, clusters, gt_col, number=0):
    label_all = []
    for index_a in clu_list:
        cluster_a_per = clusters[index_a] if number == 0 else clusters[index_a - number]
        labels = [gt_col[i[0]] for i in cluster_a_per]
        cluster_label = most_frequent(labels)

        label_all.append(cluster_label)
    return label_all


def clustersMerge1(clusters_a, clustera_name, clusters_b, clusterb_name, dataset, method="single"):
    def check_values_in_list(lst, n):
        has_less_than_n = [x for x in lst if x < n]

        has_greater_than_n = [x for x in lst if x > n]
        return has_less_than_n, has_greater_than_n

    all = len(clusters_a) + len(clusters_b)

    cluster_all = clusters_a + clusters_b
    distances = np.zeros((all, all))
    for i, a in enumerate(cluster_all):
        a_embedding = np.array([t[1] for t in a])
        j = 1
        for b in cluster_all[i + 1:]:
            b_embedding = np.array([t[1] for t in b])
            dist = calculate_linkage_distance(a_embedding, b_embedding, method=method)
            distances[i, i + j] = dist
            j += 1
    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset)

    gt_col_a = gt_clusters[clustera_name]
    gt_col_b = gt_clusters[clusterb_name]
    linkage_m = linkage(np.array(distances), method="single")
    dendrogramz = dendrogram(linkage_m)
    best_threshold = findBestSilhouetteDendro(dendrogramz, linkage_m, distances)[0]
    silhouette_avg, custom_clusters = sliced_clusters(linkage_m, best_threshold - 0.8, distances)
    # print(silhouette_avg, len(custom_clusters), custom_clusters)
    for index, cluster in custom_clusters.items():
        a_clu, b_clu = check_values_in_list(cluster, len(clusters_a) - 1)
        if len(a_clu) > 0 and len(b_clu) > 0:
            # print(a_clu, "\n", b_clu)
            cluster_labelsa = labels_found(a_clu, cluster_all, gt_col_a)
            cluster_labelsb = labels_found(b_clu, cluster_all, gt_col_b)
        # print(cluster_labelsa, "\n", cluster_labelsb)


# def clustersMerge(clusters_a,clustera_name, clusters_b,clusterb_name,dataset, method="complete"):


def groundTruthRelation(dataset):
    with open(f'datasets\{dataset}\Relationship_gt.pickle', 'rb') as handle:
        gt_relationship = pickle.load(handle)
    gt_attributes = []
    for key, value in gt_relationship.items():
        for atttr_p in value:
            a1 = f"{key[0]}.{atttr_p[0]}"
            a2 = f"{key[1]}.{atttr_p[1]}"
            gt_attributes.append((a1, a2))
    # print(gt_relationship.keys())
    # print(gt_attributes)
    return gt_relationship, gt_attributes


def embeddings_dataset(datafile_path, embedding_file, table_names, data_path):
    F = open(os.path.join(datafile_path, embedding_file), 'rb')
    if embedding_file.endswith("_column.pkl"):
        content = pickle.load(F)
        content = [(i[0], i[1][0]) for i in content]
    else:
        original_content = pickle.load(F)
        original_content_dict = {i[0]: i[1] for i in original_content}
        content = []
        for fileName in table_names:
            embedding_fileName = []
            table = pd.read_csv(os.path.join(data_path, fileName))
            for index, col in enumerate(table.columns):
                col_embed_tuple = f"{fileName[:-4]}.{col}", original_content_dict[fileName][index]
                content.append(col_embed_tuple)

    return content


def clusteringEmbedding(embedding_file, index, dataset, content, groundTruth=False, clustering="Agglomerative"):
    filename = f"{index}_colcluster_dict.pickle" if not groundTruth else f"{index}_colcluster_dictGT.pickle"
    F_cluster = open(os.path.join(os.getcwd(), "result/SILM/", dataset,
                                  "All/" + embedding_file[:-4] + "/column", filename), 'rb')
    ### TODO hard coded problem

    col_cluster = pickle.load(F_cluster)[clustering]
    clusters = list(col_cluster.values())
    # print("clusters", len(clusters))
    clusters_embedding = []
    for cluster in clusters:
        cluster_embedding = [i for i in content if i[0] in cluster]
        clusters_embedding.append((cluster_embedding))

    return clusters_embedding


def P4(hp: Namespace):
    # Read table and their corresponding lowest type
    subjectColPath = os.path.join(os.getcwd(), f"datasets/{hp.dataset}")
    data_path = os.path.join(subjectColPath, "Test")
    table_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]

    SE = subjectColumns(subjectColPath)

    # embedding of the tables
    datafile_path = os.getcwd() + "/result/embedding/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if
             fn.endswith(
                 '.pkl') and f"_{hp.embed}_" in fn]  # and 'SCT6' in fn and 'header' not in fn
    files = [fn for fn in files if not fn.endswith("subCol.pkl")]  # and 'Pretrain' in fn and 'header' in fn
    # print(files)

    # ground truth of conceptual attributes
    gt_relationship, gt_attributes = groundTruthRelation(hp.dataset)

    score_path = os.path.join(os.getcwd(),
                              f"result/P4/new/{hp.dataset}/")
    mkdir(score_path)

    # start to test finding relationship in each embedding method
    for embedding_file in files:
        #     print(embedding_file)
        content = embeddings_dataset(datafile_path, embedding_file, table_names, data_path)

        cluster_relationships = {}
        target_path = os.path.join(os.getcwd(),
                                   f"result/P4/{hp.dataset}/{embedding_file[:-4]}/")
        # Start to calculate time
        startTimeS = time.time()
        gt_clusters, ground_t, gt_cluster_dict = column_gts(hp.dataset)
        keys = list(gt_cluster_dict.keys())

        ### TODO hard coded problem
        for index, clu in enumerate(keys):

            if clu == "['Person']":
                cluster_a = clusteringEmbedding(embedding_file, index, hp.dataset, content, clustering=hp.clustering)

                for clu_j in keys[index + 1:]:
                    if clu_j == "['Place']":
                        print(clu, clu_j)
                        cluster_b = clusteringEmbedding(embedding_file, keys.index(clu_j), hp.dataset, content,
                                                        clustering=hp.clustering)
                        clustersMerge1(cluster_a, clu, cluster_b, clu_j, hp.dataset)


def MetricType(groundTruth, results):
    TP = [i for i in results if i in groundTruth]
    FP = [i for i in results if i not in groundTruth]
    FN = [i for i in groundTruth if i not in results]

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    F1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, F1score


def relationshipDiscovery(hp: Namespace):
    timing = []
    # load data, load subject column in each table
    subjectColPath = os.path.join(os.getcwd(), f"datasets/{hp.dataset}")
    data_path = os.path.join(subjectColPath, "Test")
    SE = subjectColumns(subjectColPath)
    table_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]
    # ground truth of the type of tables
    ground_truth = os.path.join(os.getcwd(), f"datasets/{hp.dataset}/groundTruth.csv")
    Ground_t = group_files(pd.read_csv(ground_truth))
    types = list(Ground_t.keys())
    # print("types",types)
    # load the embedding file of different embedding methods
    datafile_path = os.getcwd() + "/result/embedding/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if
             fn.endswith('.pkl') and f"_{hp.embed}_" in fn]  # and 'SCT6' in fn and 'header' not in fn
    files = [fn for fn in files if not fn.endswith("subCol.pkl")]  # and 'Pretrain' in fn and 'header' in fn
    # print("files",files)
    # create the folder that stores the scores of column matching using
    score_path = os.path.join(os.getcwd(), f"result/P4/{hp.dataset}/")
    mkdir(score_path)
    for embedding_file in files[0:]:
        # load the column name and corresponding vector
        print("embedding_file", embedding_file)
        F = open(os.path.join(datafile_path, embedding_file), 'rb')
        if embedding_file.endswith("_column.pkl"):
            original_content = pickle.load(F)
            content = []
            for fileName in table_names:
                embedding_fileName = []
                table = pd.read_csv(os.path.join(data_path, fileName))
                for col in table.columns:
                    embedding_col = [i[1] for i in original_content if i[0] == f"{fileName[:-4]}.{col}"][0]
                    embedding_fileName.append(embedding_col[0])
                tuple_file = fileName, np.array(embedding_fileName)
                content.append(tuple_file)
        else:
            content = pickle.load(F)
        cluster_relationships = {}
        startTimeS = time.time()
        for index, type_i in enumerate(types):
            for type_j in types[index + 1:]:
                cluster1 = Ground_t[type_i]
                cluster2 = Ground_t[type_j]
                # if type_i == 'Organization' and type_j == 'Person':
                cluster1_embedding = [i for i in content if i[0] in cluster1]
                cluster2_embedding = [i for i in content if i[0] in cluster2]
                relationship1 = entityTypeRelationship(cluster1_embedding, cluster2_embedding, hp.similarity,
                                                       SE, isEuclidean=hp.Euclidean)
                relationship2 = entityTypeRelationship(cluster2_embedding, cluster1_embedding, hp.similarity,
                                                       SE, isEuclidean=hp.Euclidean)
                if len(relationship1) > 0:
                    cluster_relationships[(type_i, type_j)] = relationship1
                if len(relationship2) > 0:
                    cluster_relationships[(type_j, type_i)] = relationship2
        endTimeS = time.time()
        timing.append({'Embedding File': embedding_file, "timing": endTimeS - startTimeS})
        gt_relationship = relationshipGT(hp.dataset)
        target_path = os.path.join(os.getcwd(),
                                   f"result/P4/{hp.dataset}/{embedding_file[:-4]}/")
        mkdir(target_path)
        mid_string = "_Base" if hp.baseline is True else ""
        file_pickle = f'Relationships_{mid_string}_{str(hp.similarity)}_{hp.embed}_por{str(hp.portion)}_Eu{hp.Euclidean}.pickle'
        attributeRelation(hp.dataset, target_path, cluster_relationships, file_pickle.split(".pickle")[0] + ".csv")
        ### Todo STOP HERE: Needs to refine the metric function
        # timing_df = pd.DataFrame(timing)
        # timing_df.to_csv(os.path.join(os.getcwd(), f"result/P4/{hp.dataset}/timing_{hp.embed}.csv"))
        result_type_pairs = list(cluster_relationships.keys())
        p, r, f1 = MetricType(list(gt_relationship[0].keys()), result_type_pairs)
        precision, recall, sort_result, TN = attributeRelationshipSearch(cluster_relationships, hp, embedding_file, SE,
                                                                         gt_relationship[1])
        attribute_pairs = [i for i in sort_result]
        storing = (cluster_relationships, sort_result, TN)
        with open(os.path.join(target_path, file_pickle), 'wb') as handle:
            pickle.dump(storing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(embedding_file, " type metric:\n",
              f"precision {p} recall: {r} F1-score: {f1}\n",
              f"attribute Precision: {precision}",
              f"attribute recall: {recall}")
        type_csv = f'TypeRelationshipScore{mid_string}_Eu{hp.Euclidean}.csv'
        attri_csv = f'AttributeRelationshipScore{mid_string}_Eu{hp.Euclidean}.csv'
        if type_csv in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, type_csv), index_col=0)
        else:
            df = pd.DataFrame(columns=['Similarity', 'Portion', 'Embedding', 'Precision', 'Recall', 'F1-score'])
            df.to_csv(os.path.join(score_path, type_csv))

        Eu = "EU_" if hp.Euclidean is True else "COS_"
        index = Eu + str(hp.similarity) + embedding_file[:-4] + str(hp.portion)
        if index not in df.index:
            new_data = {'Similarity': hp.similarity, 'Portion': hp.portion,
                        'Embedding': embedding_file[:-4], "Precision": p,
                        "Recall": r, "F1-score": f1}
            # print(new_data)
            new_row = pd.DataFrame([new_data], index=[index])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            # print(df)
            df.to_csv(os.path.join(score_path, type_csv))
        else:
            df.loc[str(hp.similarity), 'Similarity'] = hp.similarity
            df.loc[str(hp.similarity), 'Portion'] = hp.portion
            df.loc[str(hp.similarity), 'Embedding'] = embedding_file[:-4]
            df.loc[str(hp.similarity), 'Precision'] = p
            df.loc[str(hp.similarity), 'Recall'] = r
            df.loc[str(hp.similarity), 'F1-score'] = f1

            # print(df)
            df.to_csv(os.path.join(score_path, type_csv))

        if attri_csv in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, attri_csv), index_col=0)
        else:
            df = pd.DataFrame(
                columns=['Similarity', 'Portion', 'Embedding', 'Precision', 'Recall', "attribute result", "TN"])
            df.to_csv(os.path.join(score_path, attri_csv))

        if index not in df.index:
            new_data = {'Similarity': hp.similarity, 'Portion': hp.portion, 'Embedding': embedding_file[:-4],
                        "Precision": precision, "Recall": recall,
                        "attribute result": attribute_pairs, "TN": TN}
            # print(new_data)
            new_row = pd.DataFrame([new_data], index=[index])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            # print(df)
            df.to_csv(os.path.join(score_path, attri_csv))
        else:
            df.loc[str(hp.similarity), 'Similarity'] = hp.similarity
            df.loc[str(hp.similarity), 'Portion'] = hp.portion
            df.loc[str(hp.similarity), 'Embedding'] = embedding_file[:-4]
            df.loc[str(hp.similarity), 'Precision'] = precision
            df.loc[str(hp.similarity), 'Recall'] = recall
            df.loc[str(hp.similarity), 'attribute result'] = str(attribute_pairs)
            df.loc[str(hp.similarity), 'TN'] = str(TN)
            # print(df)
            df.to_csv(os.path.join(score_path, attri_csv))


def relationshipGT(dataset):
    gt = pd.read_excel(f"datasets\{dataset}\Relationship.xlsx")
    gt_dict = {}
    for index, row in gt.iterrows():
        t1, t2 = row[gt.columns[0]], row[gt.columns[2]]
        sa1, nsa2 = row[gt.columns[1]], row[gt.columns[3]]
        if (t1, t2) not in gt_dict.keys():
            gt_dict[(t1, t2)] = []
        gt_dict[(t1, t2)].append(nsa2)
    relation_attributes = []
    for table_pair, attributes in gt_dict.items():
        for attr in attributes:
            relation_attributes.append((f"{table_pair[0]}", f"{table_pair[1]}.{attr}"))
    return gt_dict, relation_attributes


def attributeRelationshipSearch(cluster_relationships, hp: Namespace, embedding_file, SE, gt_attributes):
    def check_conceptualAttri(GroundTruth, gtClusters, table, subcols, gt_clusterDict=None):
        type = [key for key, value in GroundTruth.items() if table in value][0]
        if hp.baseline is False:
            index = list(gt_clusterDict.keys()).index(type)
            checkfile = f"{index}_colcluster_dict.pickle"
            F_cluster = open(os.path.join(datafile_path, checkfile), 'rb')
            # TODO needs to change this in the table clustering -> clustering into soft coded
            col_cluster = pickle.load(F_cluster)[hp.clustering]
            gt_clusters_type = gtClusters[type]
            conceptual = find_conceptualAttri(col_cluster, table, subcols, gtclusters=gt_clusters_type)
        else:
            conceptual = find_conceptualAttri(gtClusters[type], table, subcols)
        # length = [len(gt_attribtues[type][i]) for i in conceptual]
        return conceptual

    def metric(gt_list, result_list, metric='P'):
        resultK = [i for i in result_list if i in gt_list]
        print(resultK, "\n", result_list)
        if metric == 'P':
            metric_score = len(resultK) / len(result_list) if len(result_list) > 0 else 0
        else:
            metric_score = len(resultK) / len(gt_list)
        return metric_score, resultK

    print(cluster_relationships)
    # read generated conceptual attributes of the embedding methods
    gt_clusters, ground_t, gt_cluster_dict = column_gts(hp.dataset)
    datafile_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                 "All/" + embedding_file[:-4] + "/column")
    # type of the table
    ground_truth = os.path.join(os.getcwd(), f"datasets/{hp.dataset}/groundTruth.csv")
    Ground_t = group_files_by_superclass(pd.read_csv(ground_truth))
    relationships = {}
    for type_pair, table_pair_dict in cluster_relationships.items():
        relationships[type_pair] = {}
        for table_pair, similarities in table_pair_dict.items():
            # print(table_pair, similarities)
            t1, t2 = table_pair[0], table_pair[1]
            NE_list_1, columns1, types1 = SE[t1]
            t1_subcol = [columns1[NE_list_1[0]]] if len(NE_list_1) != 0 else [columns1[0]]
            t2_cols = list(similarities.keys())
            t1_Attribute = check_conceptualAttri(Ground_t, gt_clusters, t1, t1_subcol,
                                                 gt_clusterDict=gt_cluster_dict)
            t2_Attribute = check_conceptualAttri(Ground_t, gt_clusters, t2, t2_cols,
                                                 gt_clusterDict=gt_cluster_dict)
            for Attri_i in t1_Attribute:
                for index, Attri_j in enumerate(t2_Attribute):
                    if (Attri_i, Attri_j) not in relationships[type_pair]:
                        relationships[type_pair][(Attri_i, Attri_j)] = {
                            'similarity': [list(similarities.values())[index]], "tables_RA": [t2], "tables_SA": [t1]}
                    else:
                        relationships[type_pair][(Attri_i, Attri_j)]['similarity'].append(
                            list(similarities.values())[index])
                        if t2 not in relationships[type_pair][(Attri_i, Attri_j)]['tables_RA']:
                            relationships[type_pair][(Attri_i, Attri_j)]['tables_RA'].append(t2)
                        if t1 not in relationships[type_pair][(Attri_i, Attri_j)]['tables_SA']:
                            relationships[type_pair][(Attri_i, Attri_j)]['tables_SA'].append(t1)
    overall_results = {}
    ### TODO I think this needs to simplify
    ### TODO maybe we need to move this to some other place when we simplify the matching process
    for type_pair, attribute_pair_dict in relationships.items():
        for attribute_pair, attribute_dict in attribute_pair_dict.items():
            attribute_similarities = attribute_dict["similarity"]
            type_table_RA = \
                [key for key, value in Ground_t.items() if type_pair[1] in key and attribute_pair[1] in ground_t[key]][
                    0]
            portion_attribute_RA = len(attribute_dict["tables_RA"]) / len(ground_t[type_table_RA][attribute_pair[1]])
            type_table_SA = [key for key, value in Ground_t.items() if type_pair[0] in key][0]

            portion_attribute_SA = len(attribute_dict["tables_SA"]) / len(Ground_t[type_table_SA])
            # print(attribute_pair, portion_attribute_SA)
            # print(attribute_pair,len(ground_t[type_table][attribute_pair[1]]), portion_attribute, attribute_dict )
            if portion_attribute_RA >= hp.portion: #and portion_attribute_SA >= hp.portion
                attri_combine1 = f"{type_pair[0]}"
                attri_combine2 = f"{type_pair[1]}.{attribute_pair[1]}"
                sim_attri = sum(attribute_similarities) / len(attribute_similarities)
                overall_results[(attri_combine1, attri_combine2)] = {'sim': sim_attri, 'portion_SA': type_table_SA,
                                                                     'portion_RA': portion_attribute_RA}
    precision, TN = metric(gt_attributes, overall_results)
    recall = metric(gt_attributes, overall_results, metric='R')[0]
    return precision, recall, overall_results, TN


"""for type_pair, attribute_pair_dict in relationships.items():
        for attribute_pair, attribute_similarities in attribute_pair_dict.items():
            attri_combine1 = f"{type_pair[0]}.{attribute_pair[0]}"
            attri_combine2 = f"{type_pair[1]}.{attribute_pair[1]}"
            sim_attri = sum(attribute_similarities) / len(attribute_similarities)
            overall_results[(attri_combine1, attri_combine2)] = sim_attri
    sort_results = sorted(overall_results.items(), key=lambda item: item[1])
    precision, TN = metric(gt_attributes, sort_results)
    recall = metric(gt_attributes, sort_results, metric='R')[0]"""


# TODO this needs to delete later
def find_conceptualAttribute(gt_col, type, table, attributes):
    combines = [f"{table[:-4]}.{attribute}" for attribute in attributes]
    conceptualAttributes = [gt_col[type][i] for i in combines if i in gt_col[type].keys()]
    return conceptualAttributes


def attributeRelation(dataset, target_path, cluster_relationships, file_target):
    ground_t = column_gts(dataset)[0]
    ground_truth = os.path.join(os.getcwd(), f"datasets/{dataset}/groundTruth.csv")
    Ground_t = group_files_by_superclass(pd.read_csv(ground_truth))
    subjectColPath = os.path.join(os.getcwd(), f"datasets/{dataset}")
    SE = subjectColumns(subjectColPath)
    column_relationships = []
    for type_pair, table_pair_dict in cluster_relationships.items():
        for table_pair, similarities in table_pair_dict.items():
            # print(table_pair, similarities)
            t1 = table_pair[0]
            t2 = table_pair[1]
            type_1_list = [key for key, value in Ground_t.items() if t1 in value][0]
            type_2_list = [key for key, value in Ground_t.items() if t2 in value][0]
            NE_list_1, columns1, types1 = SE[t1]
            t1_subcol = columns1[NE_list_1[0]] if len(NE_list_1) != 0 else columns1[0]
            t2_cols = list(similarities.keys())
            t2_cols_score = list(similarities.values())
            t1_conceptual = find_conceptualAttribute(ground_t, type_1_list, t1, [t1_subcol])
            if len(t1_conceptual) == 0:
                concept = t1_subcol
            else:
                concept = t1_conceptual[0]
            t2_conceptual = find_conceptualAttribute(ground_t, type_2_list, t2, t2_cols)
            for t2_col_concept in t2_conceptual:
                column_relationships.append(
                    {'Type1': type_pair[1], 'SubAttribute': concept, 'table1': t1, 'subcol': t1_subcol,
                     'T2': type_pair[0], 'Attribute': t2_col_concept, 'table2': t2,
                     'similarity': t2_cols_score[t2_conceptual.index(t2_col_concept)]})

    column_relationships = pd.DataFrame(column_relationships)
    # print(column_relationships)
    column_relationships.to_csv(os.path.join(target_path, file_target), index=False)
    # types.to_csv("datasets/WDC/clusterRelationships.csv",index=False)
