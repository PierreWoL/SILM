import ast
import os
from collections import Counter
from typing import Optional
from scipy.spatial.distance import cosine, euclidean
import experimentalData as ed
import statistics
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
# from d3l.indexing.similarity_indexes import DistributionIndex
# from d3l.input_output.dataloaders import CSVDataLoader
# from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import numpy as np
from itertools import combinations

from sklearn.manifold import TSNE
import plotly.express as px


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def rand_Index_custom(predicted_labels, ground_truth_labels):
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
                all += 1
    RI = (true_positive + true_negative) / all
    return RI


def create_or_find_indexes(data_path, threshold, embedding_mode=1):
    #  collection of tables
    """
    dataloader = CSVDataLoader(
        root_path=data_path,
        # sep=",",
        encoding='latin-1'
    )

    # NameIndex
    # if os.path.isfile(os.path.join(data_path, './name.lsh')):
    #   name_index = unpickle_python_object(os.path.join(data_path, './name.lsh'))
    # else:
    name_index = NameIndex(dataloader=dataloader, index_similarity_threshold=threshold)
    pickle_python_object(name_index, os.path.join(data_path, './name.lsh'))

   
    # FormatIndex
    # if os.path.isfile(os.path.join(data_path, './format.lsh')):
    #   format_index = unpickle_python_object(os.path.join(data_path, './format.lsh'))
    # else:
    format_index = FormatIndex(dataloader=dataloader, index_similarity_threshold=threshold)
    pickle_python_object(format_index, os.path.join(data_path, './format.lsh'))
    """
    """
    # ValueIndex
    if os.path.isfile(os.path.join(data_path, './value.lsh')):
    #   value_index = unpickle_python_object(os.path.join(data_path, './value.lsh'))
    # else:
    value_index = ValueIndex(dataloader=dataloader, index_similarity_threshold=threshold)
    pickle_python_object(value_index, os.path.join(data_path, './value.lsh'))
    """
    # DistributionIndex
    # if os.path.isfile(os.path.join(data_path, './distribution.lsh')):
    #   distribution_index = unpickle_python_object(os.path.join(data_path, './distribution.lsh'))
    # else:
    # distribution_index = DistributionIndex(dataloader=dataloader, index_similarity_threshold=threshold)
    # pickle_python_object(distribution_index, os.path.join(data_path, './distribution.lsh'))
    """
    # EmbeddingIndex
     
    embed_lsh = './embedding' + str(embedding_mode) + '.lsh'
    # if os.path.isfile(os.path.join(data_path, embed_lsh)):
    #   embeddingIndex = unpickle_python_object(os.path.join(data_path, embed_lsh))
    # else:

    embeddingIndex = EmbeddingIndex(dataloader=dataloader, mode=embedding_mode, index_similarity_threshold=threshold)
    pickle_python_object(embeddingIndex, os.path.join(data_path, embed_lsh))
    """
    return []  # distribution_index , format_index, value_index, name_index, embeddingIndex


"""def initialise_distance_matrix(dim, L, dataloader, data_path, indexes, k):
    #print(L)
    D = np.ones((dim, dim))
    T = ed.get_files(data_path)
    # Things are the same as themselves
    for i in range(dim):
        D[i, i] = 0
    for t in T:
        # qe = QueryEngine(name_index, format_index, value_index, distribution_index)
        qe = QueryEngine(*indexes)

        Neighbours = qe.table_query(table=dataloader.read_table(table_name=t),
                                    aggregator=None, k=k)
        # print(Neighbours,len(Neighbours))
        for n in Neighbours:  # index
            (name, similarities) = n  # 'car', [1.0, 0.4, 0.4, 0.0]
            if name in L and t != name:
                # print(L[t], L[name])
                # print(L[name], L[t])
                D[L[t], L[name]] = 1 - statistics.mean(similarities)
                D[L[name], L[t]] = 1 - statistics.mean(similarities)

    return D
    
    
    
def distance_matrix(data_path, index, k):
    # print(index)
    dataloader = CSVDataLoader(root_path=(data_path), encoding='latin-1')
    T = ed.get_files(data_path)
    # print(T)
    L = {}
    for i, t in enumerate(T):
        L[t] = i
    # before_distance_matrix = time()
    D = initialise_distance_matrix(len(T), L, dataloader, data_path, index, k)
    # after_distance_matrix = time()
    # print("Building distance matrix took ",{after_distance_matrix-before_distance_matrix}," sec to run.")
    Z_df = pd.DataFrame(D)
    # print(Z)
    return Z_df, T    
    """


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


"""
Useless

def cluster_discovery(parameters):
    db = DBSCAN(eps=parameters[0],
                min_samples=parameters[1],
                n_jobs=-1)
    db.fit(parameters[2])
    labels = list(db.labels_)
    l = (parameters[3], labels)
    clust = zip(*l)
    clu = list(clust)

    return clu
"""

"""
Find BIRCH clustering algorithms
"""


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


def AgglomerativeClustering_param_search(input_data, cluster_num):
    input_data = np.array(input_data, dtype=np.float32)
    score = -1
    best_model = AgglomerativeClustering()
    if cluster_num < 10:
        at_least = 2
    else:
        at_least = cluster_num - 10
    for n_clusters in range(at_least, cluster_num + 10):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        agg_clustering.fit(input_data)
        labels = agg_clustering.labels_
        if score <= silhouette_score(input_data, labels):
            score = silhouette_score(input_data, labels)
            best_model = agg_clustering
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


def most_frequent(list1, isFirst=True):
    """
    count the most frequent occurring annotated label in the cluster
    """

    count = Counter(list1)
    if isFirst is True:
        return count.most_common(1)[0][0]
    else:
        most_common_elements = count.most_common()
        max_frequency = most_common_elements[0][1]
        most_common_elements_list = [element for element, frequency in most_common_elements if
                                     frequency == max_frequency]
        return most_common_elements_list


"""
f = open(T2DV2GroundTruth, encoding='latin1', errors='ignore')
gt_CSV = pd.read_csv(f, header=None)
GroundTruth = dict(zip(gt_CSV[0].str.removesuffix(".tar.gz"), gt_CSV[1]))
gt_clusters, ground_t = ed.get_concept_files(ed.get_files(samplePath), GroundTruth)
gt_cluster = gt_CSV[1].unique()
gt_cluster_dict = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}

"""


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
        # print("Start iterating in cluster of column", column_list,labels,gt_column_label)
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
                     tables_gt: Optional[dict] = None):
    clusters_label = {}
    table_label_index = []
    false_ones = []

    gt_table_label = []
    tables = []

    overall_info = []
    overall_clustering = []


    for index, tables_list in clusterDict.items():
        labels = []
        for table in tables_list:
            if table in gtclusters.keys():
                tables.append(table)
                label = gtclusters[table]
                if type(label) is list:
                    for item_label in label:
                        labels.append(item_label)
                    gt_table_label.append(label)
                else:
                    gt_table_label.append(gtclusters_dict[label])
                    labels.append(label)
        if len(labels) == 0:
            continue
        else:
            cluster_label = most_frequent(labels, isFirst=False)

        clusters_label[index] = cluster_label
        current_ones = []

        for table in tables_list:
            if table in gtclusters.keys():
                if type(cluster_label) is list:
                    table_label_index.append(cluster_label)
                    if len(list(set(gtclusters[table]) & set(cluster_label))) == 0:
                        false_ones.append([table, cluster_label, gtclusters[table]])
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index,"False", tables_gt[table], gtclusters[table]])
                    else:
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "True", tables_gt[table], gtclusters[table]])

                else:
                    table_label_index.append(gtclusters_dict[cluster_label])
                    if gtclusters[table] != cluster_label:
                        false_ones.append([table, cluster_label, gtclusters[table]])
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table,index ,"False", tables_gt[table], gtclusters[table]])
                    else:
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "True", tables_gt[table], gtclusters[table]])

        if tables_gt is not None and folder is not None:
            purity_index = 1 - len(current_ones) / len(tables_list)
            overall_clustering.extend(current_ones)
            overall_info.append([index, cluster_label, purity_index, len(tables_list)])
            del current_ones, purity_index

    if type(gt_table_label[0]) is not list:
        metric_dict = metric_Spee(gt_table_label, table_label_index)
    else:
        metric_dict = {"random Index": rand_Index_custom(gt_table_label, table_label_index)}
    # cb_pairs = wrong_pairs(gt_table_label, table_label_index, tables, tables_gt)
    metric_dict["purity"] = 1 - len(false_ones) / len(gtclusters)

    if tables_gt is not None and folder is not None:
        df = pd.DataFrame(overall_clustering, columns=['tableName','cluster id' ,'table type', 'lowest type', 'top level type'])
        df2 = pd.DataFrame(overall_info, columns=['cluster id', 'corresponding top level type', 'cluster purity', 'size'])
        df.to_csv(os.path.join(folder,'overall_clustering.csv'), encoding='utf-8', index=False)
        df2.to_csv( os.path.join(folder, 'purityCluster.csv'), encoding='utf-8', index=False)
    return metric_dict


"""def inputData(data_path, threshold, k, embedding_mode=2):
    #print("embed mode is ", embedding_mode)
    indexes = create_or_find_indexes(data_path, threshold, embedding_mode=embedding_mode)
    Z, T = distance_matrix(data_path, indexes, k)
    #print("Z,T is ",Z,T )
    return Z, T"""


def clustering_results(input_data, tables, data_path, groundTruth, clusteringName, folderName=None):
    gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, groundTruth)
    gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(data_path, groundTruth, superclass=False)
    del ground_t0,gt_cluster_dict0
    parameters = []
    if clusteringName == "DBSCAN":
        parameters = dbscan_param_search(input_data)
    if clusteringName == "GMM":
        parameters = gaussian_m_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "Agglomerative":
        parameters = AgglomerativeClustering_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "OPTICS":
        parameters = OPTICS_param_search(input_data)
    if clusteringName == "KMeans":
        parameters = KMeans_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "BIRCH":
        parameters = BIRCH_param_search(input_data, len(gt_cluster_dict))
    clusters = cluster_discovery(parameters, tables)
    cluster_dict = cluster_Dict(clusters)

    """
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
    # table_dict = {tables[i]: input_data[i] for i in range(0, len(tables))}
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName, gt_clusters0)
    return cluster_dict, metrics_value


def clustering_hier_results(input_data, tables, gt_clusters, gt_cluster_dict, clusteringName, folderName=None,
                            filename=None):
    parameters = []
    if clusteringName == "DBSCAN":
        parameters = dbscan_param_search(input_data)
    if clusteringName == "GMM":
        parameters = gaussian_m_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "Agglomerative":
        parameters = AgglomerativeClustering_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "OPTICS":
        parameters = OPTICS_param_search(input_data)
    if clusteringName == "KMeans":
        parameters = KMeans_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "BIRCH":
        parameters = BIRCH_param_search(input_data, len(gt_cluster_dict))
    clusters = cluster_discovery(parameters, tables)
    cluster_dict = cluster_Dict(clusters)
    table_dict = None
    table_dict = {tables[i]: input_data[i] for i in range(0, len(tables))}
    # print("cluster_dict",cluster_dict)
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName, filename, table_dict)
    # print(metrics_value)
    return cluster_dict, metrics_value


def clusteringColumnResults(input_data, columns, gt_clusters, gt_cluster_dict, clusteringName, folderName=None,
                            filename=None):
    parameters = []
    if clusteringName == "DBSCAN":
        parameters = dbscan_param_search(input_data)
    if clusteringName == "GMM":
        parameters = gaussian_m_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "Agglomerative":
        parameters = AgglomerativeClustering_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "OPTICS":
        parameters = OPTICS_param_search(input_data)
    if clusteringName == "KMeans":
        parameters = KMeans_param_search(input_data, len(gt_cluster_dict))
    if clusteringName == "BIRCH":
        parameters = BIRCH_param_search(input_data, len(gt_cluster_dict))
    clusters = cluster_discovery(parameters, columns)
    cluster_dict = cluster_Dict(clusters)
    table_dict = None
    table_dict = {columns[i]: input_data[i] for i in range(0, len(columns))}
    metrics_value = evaluate_col_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName, filename)
    return cluster_dict, metrics_value


# print(clusters)
# print(len(cluster_dict), cluster_dict)
"""
groundTruth = os.getcwd() + "/datasets/open_data/gt_openData.csv"
import clustering as c
import os
data_path = os.getcwd()+"/datasets/open_data/test/"
gt_clusters, ground_t, gt_cluster_dict = c.data_classes(data_path, groundTruth)
gt_cluster_dict_sum = {cluster: len(ground_t[cluster]) for cluster in ground_t}
print(len(gt_cluster_dict))
print(gt_cluster_dict_sum)
df = pd.DataFrame(list(gt_cluster_dict_sum.items()), columns=['label', 'total number'])
print(df)
df.to_csv(os.getcwd()+"/datasets/open_data/cluster_distribution.csv",index=False)
"""

# print(GroundTruth)
"""
gt_clusters: a dictionary that mapping table name and the annotated class label
element in this dictionary: '68779923_0_3859283110041832023': 'Country'
gt_clusters == GroundTruth

ground_t is a dictionary of which key is class label while value is the list
of the table name which belonging to this cluster
like 'GolfPlayer': ['44206774_0_3810538885942465703']

truth: value of gt_clusters

gt_cluster: all the labels of tables  [a,b,c,d,e,...,f]
gt_cluster_dict: table name: index of the cluster label (corresponding to the gt_cluster label index)
{table1: index(a),...,}
"""

"""
create lsh indexes of samplePath
"""

"""
clusters_label: the label of result clusters
table_label_index: the label index of all tables
"""

"""

clusters_label = {}
table_label_index = []
false_ones = []
gt_table_label = []
for index, tables_list in cluster_dict.items():
    labels = []
    for table in tables_list:
        label = gt_clusters[table]
        gt_table_label.append(gt_cluster_dict[label])
        labels.append(label)
    cluster_label = most_frequent(labels)
    clusters_label[index] = cluster_label
    for table in tables_list:
        table_label_index.append(gt_cluster_dict[cluster_label])
        if gt_clusters[table] != cluster_label:
            false_ones.append([table + ".csv", cluster_label, gt_clusters[table]])
print("evaluation metrics is :", metric_Spee(gt_table_label, table_label_index))
# print(clusters_label, false_ones)

df = pd.DataFrame(false_ones, columns=['table name', 'result label', 'true label'])
results = []
for key in clusters_label.keys():
    results.append([key, cluster_dict[key], clusters_label[key]])
df2 = pd.DataFrame(results, columns=['cluster number', 'tables', 'label'])
baselinePath = os.getcwd() + "/result/subject_column/"
# df.to_csv(baselinePath + 'testbeta8.csv', encoding='utf-8', index=False)
# df2.to_csv(baselinePath + 'testbeta8_meta.csv', encoding='utf-8', index=False)
"""

"""
false_ones: the list of all tables 
if they have been clustered to the wrong cluster 
compared to the original label
"""
"loop: the original label index of all tables"
