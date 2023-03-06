import os
from collections import Counter
import experimentalData as ed
import statistics
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import numpy as np


def create_or_find_indexes(data_path, embedding_mode=1):
    #  collection of tables
    dataloader = CSVDataLoader(
        root_path=data_path,
        # sep=",",
        encoding='latin-1'
    )

    # NameIndex
    if os.path.isfile(os.path.join(data_path, './name.lsh')):
        name_index = unpickle_python_object(os.path.join(data_path, './name.lsh'))
    else:
        name_index = NameIndex(dataloader=dataloader)
        pickle_python_object(name_index, os.path.join(data_path, './name.lsh'))
    # FormatIndex
    if os.path.isfile(os.path.join(data_path, './format.lsh')):
        format_index = unpickle_python_object(os.path.join(data_path, './format.lsh'))
    else:
        format_index = FormatIndex(dataloader=dataloader)
        pickle_python_object(format_index, os.path.join(data_path, './format.lsh'))
        # ValueIndex
    if os.path.isfile(os.path.join(data_path, './value.lsh')):
        value_index = unpickle_python_object(os.path.join(data_path, './value.lsh'))
    else:
        value_index = ValueIndex(dataloader=dataloader)
        pickle_python_object(value_index, os.path.join(data_path, './value.lsh'))

    # DistributionIndex
    if os.path.isfile(os.path.join(data_path, './distribution.lsh')):
        distribution_index = unpickle_python_object(os.path.join(data_path, './distribution.lsh'))
    else:
        distribution_index = DistributionIndex(dataloader=dataloader)
        pickle_python_object(distribution_index, os.path.join(data_path, './distribution.lsh'))

    # EmbeddingIndex
    embed_lsh = './embedding' + str(embedding_mode) + '.lsh'
    if os.path.isfile(os.path.join(data_path, embed_lsh)):
        embeddingIndex = unpickle_python_object(os.path.join(data_path, embed_lsh))
    else:
        embeddingIndex = EmbeddingIndex(dataloader=dataloader, mode=embedding_mode)
        pickle_python_object(distribution_index, os.path.join(data_path, embed_lsh))
    return [ distribution_index, format_index,value_index,name_index, embeddingIndex]#


def initialise_distance_matrix(dim, L, dataloader, data_path, indexes, k):
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

        for n in Neighbours:  # index
            (name, similarities) = n  # 'car', [1.0, 0.4, 0.4, 0.0]

            if (name in L and t != name):
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
    for min_samples in range(1, 10):
        for xi in np.linspace(0.01, 0.1, 10):
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
    for i in range(cluster_num - 2, cluster_num + 2):
        for threshold in np.arange(start=0.1, stop=1, step=0.1):
            for branchingfactor in np.arange(start=10, stop=50, step=5):
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
    score = -1
    best_model = AgglomerativeClustering()
    for n_clusters in range(cluster_num - 2, cluster_num + 2):
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
    n_components_range = range(cluster_num - 3, cluster_num + 3)
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
    print(a, b, c, d)
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
    return {"MI": MI, "NMI": NMI, "AMI": AMI, "random score": rand_score,
            "ARI": adjusted_random_score, "FMI": FMI}


def most_frequent(list1):
    """
    count the most frequent occurring annotated label in the cluster
    """

    count = Counter(list1)
    return count.most_common(1)[0][0]


"""
f = open(T2DV2GroundTruth, encoding='latin1', errors='ignore')
gt_CSV = pd.read_csv(f, header=None)
GroundTruth = dict(zip(gt_CSV[0].str.removesuffix(".tar.gz"), gt_CSV[1]))
gt_clusters, ground_t = ed.get_concept_files(ed.get_files(samplePath), GroundTruth)
gt_cluster = gt_CSV[1].unique()
gt_cluster_dict = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}

"""


def data_classes(data_path, groundTruth_file):
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
    # print(ground_truth_df)
    # print(ground_truth_df.iloc[0,0])
    test_table = []
    dict_gt = {}
    if ground_truth_df.iloc[0, 0].endswith(".tar.gz"):
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".tar.gz"), ground_truth_df.iloc[:, 1]))
    if ground_truth_df.iloc[0, 0].endswith(".json"):
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".json"), ground_truth_df.iloc[:, 1]))
    if ground_truth_df.iloc[0, 0].endswith(".csv"):
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"), ground_truth_df.iloc[:, 1]))
        test_table = list(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"))
    test_table2 = {}.fromkeys(test_table).keys()
    # print(dict_gt.keys())
    gt_clusters, ground_t = ed.get_concept_files(ed.get_files(data_path), dict_gt)
    gt_cluster = pd.Series(gt_clusters.values()).unique()
    gt_cluster_dict = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
    # print(gt_clusters, ground_t, gt_cluster_dict,len(gt_clusters))
    # gt_cluster_dict{cluster:}
    print(ground_t, gt_cluster_dict, len(gt_clusters))
    return gt_clusters, ground_t, gt_cluster_dict


def cluster_Dict(clusters_list):
    cluster_dictionary = {}
    for k, v in clusters_list:
        if cluster_dictionary.get(v) is None:
            cluster_dictionary[v] = []
        cluster_dictionary[v].append(k)
    return cluster_dictionary


def wrong_pairs(labels_true, labels_pred, Tables, gtclusters_dict):
    c = []
    b = []
    for i in range(len(labels_pred)):
        for j in range(i + 1, len(labels_pred)):
            # b是在预测相同clusters中但ground truth 不同
            if labels_pred[i] == labels_pred[j] and labels_true[i] != labels_true[j]:
                b.append({i: (Tables[i], Tables[j])})
            # c是在ground truth相同clusters中但预测不同clusters
            if labels_pred[i] != labels_pred[j] and labels_true[i] == labels_true[j]:
                c.append({j: (Tables[i], Tables[j])})
    cb = pd.concat([pd.Series(c), pd.Series(b)], axis=1)
    return cb


def evaluate_cluster(gtclusters, gtclusters_dict, clusterDict: dict, folder=None, filename=None):
    clusters_label = {}
    table_label_index = []
    false_ones = []
    gt_table_label = []
    tables = []
    for index, tables_list in clusterDict.items():
        labels = []
        for table in tables_list:
            tables.append(table)
            label = gtclusters[table]
            gt_table_label.append(gtclusters_dict[label])
            labels.append(label)
        cluster_label = most_frequent(labels)
        clusters_label[index] = cluster_label
        for table in tables_list:
            table_label_index.append(gtclusters_dict[cluster_label])
            if gtclusters[table] != cluster_label:
                false_ones.append([table + ".csv", cluster_label, gtclusters[table]])
    metric_dict = metric_Spee(gt_table_label, table_label_index)
    cb_pairs = wrong_pairs(gt_table_label, table_label_index, tables, gtclusters_dict)
    metric_dict["purity"] = 1 - len(false_ones) / len(gtclusters)
    if folder is not None and filename is not None:
        df = pd.DataFrame(false_ones, columns=['table name', 'result label', 'true label'])
        results = []
        for key in clusters_label.keys():
            results.append([key, clusterDict[key], clusters_label[key]])
        df2 = pd.DataFrame(results, columns=['cluster number', 'tables', 'label'])
        # baselinePath = os.getcwd() + "/result/subject_column/"
        df.to_csv(folder + filename + 'K3_lshSBert.csv', encoding='utf-8', index=False)
        df2.to_csv(folder + filename + 'K3_lshSBert_meta.csv', encoding='utf-8', index=False)
        cb_pairs.to_csv(folder + filename + 'K3_lshSBert_cb.csv', encoding='utf-8', index=False)
    return metric_dict


def inputData(data_path, k, embedding_mode=2):
    print("embed mode is ", embedding_mode)
    indexes = create_or_find_indexes(data_path, embedding_mode=embedding_mode)
    Z, T = distance_matrix(data_path, indexes, k)
    return Z, T


def clustering_results(input_data, tables, data_path, groundTruth, clusteringName, folderName, filename):
    # clustering的ground truth
    gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, groundTruth)
    print(len(gt_clusters), len(ground_t))
    # 实现LSH indexes 为数据
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
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folderName, filename)
    print(metrics_value)
    return metrics_value


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
