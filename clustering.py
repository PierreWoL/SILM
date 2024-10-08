import ast
import os
import time
from collections import Counter
from Utils import most_frequent
from typing import Optional
from scipy.spatial.distance import cosine, euclidean
import experimentalData as ed
import statistics
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics

from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import numpy as np
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object




def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def rand_Index_custom(predicted_labels, ground_truth_labels):
    print(len(predicted_labels))
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    all = 0
    for i in range(len(predicted_labels)):
        i_predict = predicted_labels[i]
        i_true = ground_truth_labels[i]
        for j in range(i, len(predicted_labels)):
            if i != j:
                all += 1
                j_predict = predicted_labels[j]
                j_true = ground_truth_labels[j]
                jaccard_sim_predict = jaccard_similarity(set(i_predict), set(j_predict))
                jaccard_sim_true = jaccard_similarity(set(i_true), set(j_true))
                if jaccard_sim_predict > 0 and jaccard_sim_true > 0:
                    true_positive += 1
                # print(all,i_predict,j_predict, "and ground truth", i_true,j_true )
                elif jaccard_sim_predict == 0 and jaccard_sim_true == 0:
                    true_negative += 1
                elif jaccard_sim_predict > 0 and jaccard_sim_true == 0:
                    # print(all, i_predict,j_predict, "and ground truth", i_true,j_true )
                    false_positive += 1
                elif jaccard_sim_predict == 0 and jaccard_sim_true > 0:
                    false_negative += 1

    print(f"TP {true_positive}, TN {true_negative} all {all}")
    RI = (true_positive + true_negative) / all
    return RI


def create_or_find_indexes(data_path, threshold, LM, subjectCol=False):
    #  collection of tables

    dataloader = CSVDataLoader(
        root_path=data_path,
        subjectCol=subjectCol,
        encoding='latin1'

    )

    # NameIndex
    name_lsh = os.path.join(data_path, f'./name_{str(LM)}.lsh') if subjectCol is False \
        else os.path.join(data_path, f'./name_SubCol_{str(LM)}.lsh')
    if os.path.isfile(name_lsh):
        name_index = unpickle_python_object(name_lsh)
    else:
        name_index = NameIndex(dataloader=dataloader, index_similarity_threshold=threshold, model=LM)
        pickle_python_object(name_index, name_lsh)

    # FormatIndex
    format_lsh = os.path.join(data_path, './format.lsh') if subjectCol is False \
        else os.path.join(data_path, './format_SubCol.lsh')
    if os.path.isfile(format_lsh):
        format_index = unpickle_python_object(format_lsh)
    else:
        format_index = FormatIndex(dataloader=dataloader, index_similarity_threshold=threshold)
        pickle_python_object(format_index, format_lsh)

    # ValueIndex
    value_lsh = os.path.join(data_path, './value.lsh') if subjectCol is False \
        else os.path.join(data_path, './value_SubCol.lsh')
    if os.path.isfile(value_lsh):
        value_index = unpickle_python_object(value_lsh)
    else:
        value_index = ValueIndex(dataloader=dataloader, index_similarity_threshold=threshold)
        pickle_python_object(value_index, value_lsh)

    # DistributionIndex
    distribution_lsh = os.path.join(data_path, './distribution.lsh') if subjectCol is False \
        else os.path.join(data_path, './distribution_SubCol.lsh')
    if os.path.isfile(distribution_lsh):
        distribution_index = unpickle_python_object(distribution_lsh)
    else:
        distribution_index = DistributionIndex(dataloader=dataloader, index_similarity_threshold=threshold)
        pickle_python_object(distribution_index, distribution_lsh)

    # EmbeddingIndex
    embed_name = './embedding_' + str(LM) + '.lsh' if subjectCol is False \
        else './embedding_' + str(LM) + '_SubCol.lsh'
    embedding_lsh = os.path.join(data_path, embed_name) if subjectCol is False \
        else os.path.join(data_path, embed_name)

    if os.path.isfile(embedding_lsh):
        embeddingIndex = unpickle_python_object(embedding_lsh)
    else:
        embeddingIndex = EmbeddingIndex(dataloader=dataloader, model=LM,
                                        index_similarity_threshold=threshold)
        pickle_python_object(embeddingIndex, embedding_lsh)

    return [distribution_index, format_index, value_index, name_index, embeddingIndex]  #


def initialise_distance_matrix(dataloader, data_path, indexes, k, column=False):
    # print(L)
    T = ed.get_files(data_path)
    columns = []
    if column is True:
        for t in T:
            table = dataloader.read_table(table_name=t)
            column_t = table.columns
            for col in column_t:
                columns.append((f"{t}.{col}", table[col]))
        dim = len(columns)
        D = np.ones((dim, dim))
        np.fill_diagonal(D, 0)
        L = {tuple[0]: index for index, tuple in enumerate(columns)}
        for col_tuple in columns:
            col_name, column = col_tuple
            qe = QueryEngine(*indexes)
            Neighbours = qe.column_query(col_tuple[1], aggregator=None, k=k)
            for n in Neighbours:
                (name, similarities) = n
                # print(name, similarities)
                if name in L and col_name != name:
                    # print(L[t], L[name])
                    # print(L[name], L[t])

                    D[L[col_name], L[name]] = 1 - statistics.mean([float(sim) for sim in similarities])
                    D[L[name], L[col_name]] = 1 - statistics.mean([float(sim) for sim in similarities])
                    # print(L[col_name], L[name],similarities, D[L[col_name], L[name]] )

        cols = [i[0] for i in columns]
        # for row in D:
        # print(row)
        return D, cols

    else:
        dim = len(T)
        D = np.ones((dim, dim))
        np.fill_diagonal(D, 0)
        L = {t: i for i, t in enumerate(T)}
        #  Neighbours = qe.column_query(column=,aggregator=None, k=k)
        for t in T:
            qe = QueryEngine(*indexes)
            Neighbours = qe.table_query(table=dataloader.read_table(table_name=t),
                                        aggregator=None, k=k)
            for n in Neighbours:
                (name, similarities) = n
                if name in L and t != name:
                    # print(L[t], L[name])
                    # print(L[name], L[t])
                    D[L[t], L[name]] = 1 - statistics.mean(similarities)
                    D[L[name], L[t]] = 1 - statistics.mean(similarities)

        return D, T


def distance_matrix(data_path, index, k, column=False):
    # print(index)
    dataloader = CSVDataLoader(root_path=(data_path), encoding='latin1')
    # before_distance_matrix = time()
    D, T = initialise_distance_matrix(dataloader, data_path, index, k, column=column)
    # after_distance_matrix = time()
    # print("Building distance matrix took ",{after_distance_matrix-before_distance_matrix}," sec to run.")
    Z_df = pd.DataFrame(D)
    return Z_df, T


def inputData(data_path, threshold, k, LM, subjectCol=False, column=False):
    # print("embed mode is ", embedding_mode)
    if subjectCol is False:
        if column is True:
            ZT_Files = os.path.join(data_path, f"D3L_{LM}_column.pkl")
        else:
            ZT_Files = os.path.join(data_path, f"D3L_{LM}.pkl")
    else:
        ZT_Files = os.path.join(data_path, f"D3L_subCol_{LM}.pkl")
    if os.path.isfile(ZT_Files):
        Z, T = unpickle_python_object(ZT_Files)
    else:
        indexes = create_or_find_indexes(data_path, threshold, LM=LM,
                                         subjectCol=subjectCol)
        Z, T = distance_matrix(data_path, indexes, k, column=column)
        # print("Z,T is ", Z, T)
        file = (Z, T)
        pickle_python_object(file, ZT_Files)
    return Z, T


def dbscan_param_search(input_data):
    # Defining the list of hyperparameters to try
    eps_list = np.arange(start=0.1, stop=5, step=0.1)
    min_sample_list = np.arange(start=2, stop=25, step=1)
    score = -1
    best_dbscan = DBSCAN()
    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data = pd.DataFrame()
    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:
            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(input_data)
            labels = db.labels_
            # print(labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                if score <= silhouette_score(input_data, labels):
                    score = silhouette_score(input_data, labels)
                    best_dbscan = db
            else:
                continue
    return best_dbscan, best_dbscan.labels_


def OPTICS_param_search(input_data):
    score = -1
    best_optics = OPTICS()
    for min_samples in range(2, 10):
        for xi in np.linspace(0.1, 1, 10):
            optics = OPTICS(min_samples=min_samples, xi=xi)
            optics.fit(input_data)
            labels = optics.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                if score <= silhouette_score(input_data, labels):
                    score = silhouette_score(input_data, labels)
                    best_optics = optics
            else:
                continue
    return best_optics, best_optics.labels_


def BIRCH_param_search(input_data, cluster_num):
    score = -1
    best_birch = Birch()
    if cluster_num < 5:
        at_least = 1
    else:
        at_least = cluster_num - 3
    for i in range(at_least, cluster_num + 3):
        for threshold in np.arange(start=0.1, stop=0.8, step=0.1):
            for branchingfactor in np.arange(start=2, stop=10, step=2):
                birch = Birch(n_clusters=i, threshold=threshold, branching_factor=branchingfactor)
                birch.fit(input_data)
                labels = birch.predict(input_data)
                if score <= silhouette_score(input_data, labels):
                    score = silhouette_score(input_data, labels)
                    best_birch = birch
    return best_birch, best_birch.predict(input_data)


"""
The following is a AgglomerativeClustering
"""


def KMeans_param_search(input_data, cluster_num):
    score = -1
    best_model = KMeans()
    for i in range(cluster_num - 3, cluster_num + 3):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init='auto', random_state=0)
        kmeans.fit(input_data)
        labels = kmeans.labels_
        if score <= silhouette_score(input_data, labels):
            score = silhouette_score(input_data, labels)
            best_model = kmeans
    return best_model, best_model.labels_


def AgglomerativeClustering_param_search(input_data, cluster_num_min, cluster_num_max):
    input_data = np.array(input_data, dtype=np.float32)
    score = -1
    best_model = AgglomerativeClustering()
    # at_least = math.ceil(cluster_num // 4 * 3) + 2
    for n_clusters in range(cluster_num_min, cluster_num_max, 1):  # math.ceil(2.5* cluster_num), 3* cluster_num+10
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        agg_clustering.fit(input_data)
        labels = agg_clustering.labels_
        if score <= silhouette_score(input_data, labels):
            score = silhouette_score(input_data, labels)
            best_model = agg_clustering
    print(best_model.n_clusters, score)
    return best_model, best_model.labels_


def gaussian_m_param_search(input_data, cluster_num):
    lowest_bic = np.infty
    bic = []
    if cluster_num < 3:
        at_least = 0
    else:
        at_least = cluster_num - 3
    n_components_range = range(at_least, cluster_num + 3)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a GMM model with the specified number of components and covariance type
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
            gmm.fit(input_data)
            # Calculate the BIC score for the fitted GMM
            bic.append(gmm.bic(input_data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm, best_gmm.predict(input_data)


def cluster_discovery(parameters, tableNames):
    if not parameters:
        print("parameters are invalid! Check the code.")
        return None
    labels = parameters[1]
    l = (tableNames, labels)
    clust = zip(*l)
    clu = list(clust)
    return clu


def SMC(labels_true, labels_pred):
    distance_matrix = pairwise_distances([labels_pred], [labels_true], metric='hamming')
    # Get the number of matching pairs (a)
    a = len(labels_pred) - distance_matrix.sum()
    # Get the number of pairs that are in the same cluster in the predicted labels
    # but in different clusters in the true labels (b)
    b = 0
    for i in range(len(labels_pred)):
        for j in range(i + 1, len(labels_pred)):
            if labels_pred[i] == labels_pred[j] and labels_true[i] != labels_true[j]:
                b += 1
    # Get the number of pairs that are in different clusters in the predicted
    # labels but in the same cluster in the true labels (c)
    c = 0
    for i in range(len(labels_pred)):
        for j in range(i + 1, len(labels_pred)):
            if labels_pred[i] != labels_pred[j] and labels_true[i] == labels_true[j]:
                c += 1
    # Get the number of pairs that are in different clusters in both the predicted and true labels (d)
    d = (len(labels_pred) * (len(labels_pred) - 1) // 2) - (a + b + c)
    # Calculate the SMC
    # print(a, b, c, d)
    smc = a / (a + b + c + d)
    return smc


def metric_Spee(labels_true, labels_pred):
    MI = metrics.mutual_info_score(labels_true, labels_pred)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    rand_score = metrics.rand_score(labels_true, labels_pred)
    adjusted_random_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    FMI = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    # smc = SMC(labels_true, labels_pred)
    return {"MI": MI, "NMI": NMI, "AMI": AMI, "random Index": rand_score,
            "ARI": adjusted_random_score, "FMI": FMI}


"""
def most_frequent(list1, isFirst=True):
   
    count = Counter(list1)
    if isFirst is True:
        return count.most_common(1)[0][0]
    else:
        most_common_elements = count.most_common()
        max_frequency = most_common_elements[0][1]
        most_common_elements_list = [element for element, frequency in most_common_elements if
                                     frequency == max_frequency]
        return most_common_elements_list"""


def most_frequent_list(nested_list):
    tuples_list = [tuple(sublist) for sublist in nested_list]
    count = Counter(tuples_list)
    most_common_tuple = count.most_common(1)[0][0]
    most_common_list = list(most_common_tuple)
    return most_common_list


def data_classes(data_path, groundTruth_file, superclass=True, Nochange=False):
    """
    return three dict
    Parameters
    ----------
    data_path: the path where data stores
    groundTruth_file: table class csv files
    Returns
    -------
    gt_clusters: tablename:corresponding label dictionary. e.g.{'a': 'Book', 'b': 'Newspaper',...}
    ground_t: label and its tables. e.g. {'Book': ['a'], 'Newspaper': ['b', 'c'],...}
    gt_cluster_dict: dictionary of index: label
    like {'Political Party': 0, 'swimmer': 1, ...}
    """
    gt_file = open(groundTruth_file, errors='ignore')
    ground_truth_df = pd.read_csv(gt_file)
    test_table = []
    ground_truth_df = ground_truth_df.dropna()
    dict_gt = {}
    dict_gt0 = {}
    if superclass:
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"), ground_truth_df.iloc[:, 2]))
    else:
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"), ground_truth_df.iloc[:, 1]))

    # dict_gt = {key: ast.literal_eval(value) for key, value in dict_gt.items() if value != " " and "[" in value}
    dict_gt0 = {}
    for key, value in dict_gt.items():
        if value != " ":
            if "[" in value:
                dict_gt0[key] = ast.literal_eval(value)
            else:
                dict_gt0[key] = value
    dict_gt = dict_gt0
    # print(f"dict_gt {dict_gt}")
    test_table2 = {}.fromkeys(test_table).keys()

    gt_clusters, ground_t = ed.get_concept_files(ed.get_files(data_path), dict_gt, Nochange=Nochange)
    # print(gt_clusters.values())
    if type(list(gt_clusters.values())[0]) is list:
        if Nochange is False:
            gt_cluster_dict = {}
            for list_type in gt_clusters.values():
                for item in list_type:
                    if item not in gt_cluster_dict.keys():
                        gt_cluster_dict[item] = len(gt_cluster_dict)
        else:
            gt_cluster_dict = {}
            for cluster in list(gt_clusters.values()):
                set_cluster = str(cluster)
                if set_cluster not in gt_cluster_dict.keys():
                    gt_cluster_dict[set_cluster] = len(gt_cluster_dict)

    else:
        gt_cluster = pd.Series(gt_clusters.values()).unique()
        gt_cluster_dict = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
    return gt_clusters, ground_t, gt_cluster_dict


def cluster_Dict(clusters_list):
    cluster_dictionary = {}
    for k, v in clusters_list:
        if cluster_dictionary.get(v) is None:
            cluster_dictionary[v] = []
        cluster_dictionary[v].append(k)
    return cluster_dictionary


def wrong_pairs(labels_true, labels_pred, Tables, tables: Optional[dict] = None):
    c = []
    b = []
    for i in range(len(labels_pred)):
        for j in range(i + 1, len(labels_pred)):
            if labels_pred[i] == labels_pred[j] and labels_true[i] != labels_true[j]:
                if tables is not None:
                    cosine_sim = 1 - cosine(tables[Tables[i]], tables[Tables[j]])
                    euclidean_distance = euclidean(tables[Tables[i]], tables[Tables[j]])
                    b.append({i: (Tables[i], Tables[j], cosine_sim, euclidean_distance)})
                else:
                    b.append({i: (Tables[i], Tables[j])})

            if labels_pred[i] != labels_pred[j] and labels_true[i] == labels_true[j]:
                if tables is not None:
                    cosine_sim = 1 - cosine(tables[Tables[i]], tables[Tables[j]])
                    euclidean_distance = euclidean(tables[Tables[i]], tables[Tables[j]])
                    c.append({i: (Tables[i], Tables[j], cosine_sim, euclidean_distance)})
                else:
                    c.append({j: (Tables[i], Tables[j])})
    cb = pd.concat([pd.Series(c), pd.Series(b)], axis=1)
    return cb


def evaluate_col_cluster(gtclusters, gtclusters_dict, clusterDict: dict, folder=None, filename=None):
    clusters_label = {}
    column_label_index = []
    false_ones = []
    gt_column_label = []
    columns = []
    columns_ref = []
    for index, column_list in clusterDict.items():
        labels = []
        for column in column_list:
            if column in gtclusters.keys():
                columns.append(column)
                label = gtclusters[column]
                if type(label) is list:
                    for item_label in label:
                        labels.append(item_label)
                    gt_column_label.append(label)
                else:
                    gt_column_label.append(gtclusters_dict[label])
                    labels.append(label)

        if len(labels) == 0:
            continue
        else:
            cluster_label = most_frequent(labels)
        clusters_label[index] = cluster_label
        # print(clusters_label[index])

        false_cols = []
        for column in column_list:
            if column in gtclusters.keys():
                column_label_index.append(gtclusters_dict[cluster_label])
                if gtclusters[column] != cluster_label:
                    false_cols.append(column)
                    false_ones.append(column)

            columns_ref.append([column_list, cluster_label, false_cols])
        # print(1 - len(false_cols) / len(column_list),len(false_cols), len(column_list))
    print(gt_column_label)
    if type(gt_column_label[0]) is not list:
        metric_dict = metric_Spee(gt_column_label, column_label_index)
    else:
        metric_dict = {"random Index": rand_Index_custom(gt_column_label, column_label_index)}
    # cb_pairs = wrong_pairs(gt_table_label, table_label_index, tables, tables_gt)
    metric_dict["purity"] = 1 - len(false_ones) / len(column_label_index)

    # print(metric_dict)
    if folder is not None and filename is not None:
        # df = pd.DataFrame(false_ones, columns=['table name', 'result label', 'true label'])
        if columns:
            df_cols = pd.DataFrame(columns_ref, columns=['resultCols', 'result label', 'false_cols'])
            df_cols.to_csv(os.path.join(folder, filename + 'cols_results.csv'), encoding='utf-8', index=False)
    return metric_dict


def evaluate_cluster(gtclusters, gtclusters_dict, clusterDict: dict, folder=None,
                     tables_gt: Optional[dict] = None):  # , graph = None
    clusters_label = {}
    table_label_index = []
    false_ones = []

    gt_table_label = []
    tables = []

    overall_info = []
    overall_clustering = []
    ave_consistency = 0

    for index, tables_list in clusterDict.items():
        labels = []
        for table in tables_list:
            if table in gtclusters.keys():
                tables.append(table)
                label = gtclusters[table]
                if type(label) is list:
                    labels.append(label)
                    gt_table_label.append(label)
                else:

                    labels.append([label])
                    gt_table_label.append(gtclusters_dict[label])

        if len(labels) == 0:
            continue
        else:
            cluster_label = most_frequent_list(labels)
            # print(cluster_label)

        clusters_label[index] = cluster_label
        current_ones = []

        for table in tables_list:
            if table in gtclusters.keys():
                if type(cluster_label) is list:
                    table_label_index.append(cluster_label)
                    if len(list(set(gtclusters[table]) & set(cluster_label))) == 0:
                        false_ones.append([table, cluster_label, gtclusters[table]])
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "False", tables_gt[table], gtclusters[table]])
                    else:
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "True", tables_gt[table], gtclusters[table]])
                else:
                    table_label_index.append(gtclusters_dict[cluster_label])
                    if gtclusters[table] != cluster_label:
                        false_ones.append([table, cluster_label, gtclusters[table]])
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "False", tables_gt[table], gtclusters[table]])
                    else:
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "True", tables_gt[table], gtclusters[table]])
        lowest_gt = [i[3] for i in current_ones]
        # consistency_score = consistency_of_cluster(graph, lowest_gt)
        # ave_consistency += consistency_score
        if tables_gt is not None and folder is not None:
            falses = [i for i in current_ones if i[2] == "False"]
            purity_index = 1 - len(falses) / len(tables_list)
            overall_clustering.extend(current_ones)
            # overall_info.append([index, cluster_label, purity_index, consistency_score, len(tables_list)])
            overall_info.append([index, cluster_label, purity_index, len(tables_list)])

    if type(gt_table_label[0]) is not list:
        metric_dict = metric_Spee(gt_table_label, table_label_index)
    else:
        metric_dict = {"Random Index": rand_Index_custom(gt_table_label, table_label_index)}

    # cb_pairs = wrong_pairs(gt_table_label, table_label_index, tables, tables_gt)
    metric_dict["Purity"] = 1 - len(false_ones) / len(gtclusters)
    # metric_dict["Average cluster consistency score"] = ave_consistency / len(clusterDict)

    if tables_gt is not None and folder is not None:
        df = pd.DataFrame(overall_clustering,
                          columns=['tableName', 'cluster id', 'table type', 'lowest type', 'top level type'])
        df2 = pd.DataFrame(overall_info,
                           columns=['Cluster id', 'Corresponding top level type', 'cCluster purity', 'Size'])
        # 'Consistency of cluster', 'Size'])
        df.to_csv(os.path.join(folder, 'overall_clustering.csv'), encoding='utf-8', index=False)
        df2.to_csv(os.path.join(folder, 'purityCluster.csv'), encoding='utf-8', index=False)
        del df, df2
    return metric_dict


def clustering(input_data, data_names, number_estimate, clustering_method, max=None):
    parameters = []
    if clustering_method == "DBSCAN":
        parameters = dbscan_param_search(input_data)
    if clustering_method == "GMM":
        parameters = gaussian_m_param_search(input_data, number_estimate)
    if clustering_method == "Agglomerative":
        parameters = AgglomerativeClustering_param_search(input_data, number_estimate, max)
    if clustering_method == "OPTICS":
        parameters = OPTICS_param_search(input_data)
    if clustering_method == "KMeans":
        parameters = KMeans_param_search(input_data, number_estimate)
    if clustering_method == "BIRCH":
        parameters = BIRCH_param_search(input_data, number_estimate)
    clusters = cluster_discovery(parameters, data_names)
    cluster_dict = cluster_Dict(clusters)
    return cluster_dict


def clustering_results(input_data, tables, data_path, clusteringName, groundTruth=None, folderName=None,
                       numEstimate=0):  # , graph = None
    star_time = time.time()

    number_estimate = numEstimate  # len(gt_cluster_dict) if numEstimate==0 else numEstimate
    # print(number_estimate)

    min = number_estimate
    max = 3 * number_estimate
    # cluster_dict = clustering(input_data, tables, min, clusteringName, max=max)
    cluster_dict = clustering(input_data, tables, min, clusteringName, max=max)
    print(cluster_dict)
    end_time = time.time()
    time_difference_run = end_time - star_time
    if groundTruth is None:
        return cluster_dict, {"Clustering time": time_difference_run}

    star_time_eva = time.time()
    gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, groundTruth)
    gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(data_path, groundTruth, superclass=False)
    del ground_t0, gt_cluster_dict0
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName,
                                     gt_clusters0)  # ,graph = graph
    end_time_eva = time.time()

    time_difference_eva = end_time_eva - star_time_eva
    metrics_value["Clustering time"] = time_difference_run
    metrics_value["Evaluation time"] = time_difference_eva
    return cluster_dict, metrics_value


"""
   # THE FOLLOWING IS TO USE TSNE to show 2D figure of existing clusters
   tsne = TSNE(n_components=2, perplexity=30, random_state=0)
   transformed_data = tsne.fit_transform(input_data)
   df_display = pd.DataFrame(transformed_data, columns=['Component 1', 'Component 2'])
   df_display['name'] = tables

   df_display['label'] = [str(gt_clusters[i]) for i in tables]
   df_display['cluster'] = [f"cluster_{i[1]}" for i in clusters]

   fig = px.scatter(df_display, x='Component 1', y='Component 2', text='name', hover_data=['cluster','label','name'], color='label')
   fig.update_traces(marker=dict(size=12),
                     selector=dict(mode='markers+text'))
   fig.write_html("output_plot.html")

   fig = px.scatter(df_display, x='Component 1', y='Component 2', text='name', hover_data=['cluster', 'label', 'name'],
                    color='cluster')
   fig.update_traces(marker=dict(size=12),
                     selector=dict(mode='markers+text'))

   fig.write_html("output_plot2.html")"""


def clusteringColumnResults(input_data, columns, gt_clusters, gt_cluster_dict, clusteringName, folderName=None,
                            filename=None):
    star_time = time.time()
    number_estimate = len(gt_cluster_dict) // 2
    min = number_estimate
    max = 2 * number_estimate
    cluster_dict = clustering(input_data, columns, min, clusteringName, max=max)
    end_time = time.time()
    time_difference_run = end_time - star_time

    table_dict = None
    table_dict = {columns[i]: input_data[i] for i in range(0, len(columns))}
    star_time_eva = time.time()
    metrics_value = evaluate_col_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName, filename)
    end_time_eva = time.time()
    time_difference_eva = end_time_eva - star_time_eva
    metrics_value["Clustering time"] = time_difference_run
    metrics_value["Evaluation time"] = time_difference_eva
    return cluster_dict, metrics_value
