import os
from collections import Counter
import experimentalData as ed
import statistics
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from sklearn.metrics import pairwise_distances


def create_or_find_indexes(data_path):
    #  collection of tables
    dataloader = CSVDataLoader(
        root_path=(data_path),
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
    if os.path.isfile(os.path.join(data_path, './embedding.lsh')):
        embeddingIndex = unpickle_python_object(os.path.join(data_path, './embedding.lsh'))
    else:
        embeddingIndex = EmbeddingIndex(dataloader=dataloader)
        pickle_python_object(distribution_index, os.path.join(data_path, './embedding.lsh'))
    return [embeddingIndex, name_index, distribution_index, value_index, format_index]  #


def initialise_distance_matrix(dim, L, dataloader, data_path, indexes):
    D = np.ones((dim, dim))
    T = ed.get_files(data_path)

    # Things are the same as themselves
    for i in range(dim):
        D[i, i] = 0

    for t in T:
        # qe = QueryEngine(name_index, format_index, value_index, distribution_index)
        qe = QueryEngine(*indexes)

        Neighbours = qe.table_query(table=dataloader.read_table(table_name=t),
                                    aggregator=None, k=3)

        for n in Neighbours:  # index
            (name, similarities) = n  # 'car', [1.0, 0.4, 0.4, 0.0]

            if (name in L and t != name):
                D[L[t], L[name]] = 1 - statistics.mean(similarities)
                D[L[name], L[t]] = 1 - statistics.mean(similarities)

    return D


def dbscan_param_search(data_path, indexes):
    # print(indexes)
    dataloader = CSVDataLoader(root_path=(data_path), encoding='latin-1')
    T = ed.get_files(data_path)
    # print(T)
    L = {}
    for i, t in enumerate(T):
        L[t] = i
    # before_distance_matrix = time()
    D = initialise_distance_matrix(len(T), L, dataloader, data_path, indexes)
    # after_distance_matrix = time()
    # print("Building distance matrix took ",{after_distance_matrix-before_distance_matrix}," sec to run.")
    Z = pd.DataFrame(D)
    # print(Z)
    # Defining the list of hyperparameters to try
    eps_list = np.arange(start=0.1, stop=5, step=0.1)
    min_sample_list = np.arange(start=2, stop=25, step=1)

    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data = pd.DataFrame()
    for eps_trial in eps_list:

        for min_sample_trial in min_sample_list:
            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(Z)

            labels = db.labels_
            # print(labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                sil_score = metrics.silhouette_score(Z, labels)
            else:
                continue
            trial_parameters = "eps:" + str(eps_trial.round(1)) + " min_sample :" + str(min_sample_trial)

            silhouette_scores_data = silhouette_scores_data.append(
                pd.DataFrame(data=[[sil_score, eps_trial.round(1), min_sample_trial]],
                             columns=["score", "eps", "min_sample"]))

    # Finding out the best hyperparameters with highest Score
    par = silhouette_scores_data.sort_values(by=['score'], axis=0, ascending=False).head(1)
    eps = par.iloc[0][1]
    min_sample = par.iloc[0][2]
    return eps, int(min_sample), Z, T


def cluster_discovery(data_path, parameters):
    db = DBSCAN(eps=parameters[0],
                min_samples=parameters[1],
                n_jobs=-1)
    db.fit(parameters[2])
    labels = list(db.labels_)
    l = (parameters[3], labels)
    clust = zip(*l)
    clu = list(clust)

    return clu


T2DV2Path = os.getcwd() + "/T2DV2/"
# samplePath = os.getcwd() + "/T2DV2/test/"
samplePath = os.getcwd() + "/result/subject_column/test/"
# get_random_train_data(T2DV2Path, samplePath, 0.2)
# Concepts = get_concept(WDCFilePath)
T2DV2GroundTruth = os.getcwd() + "/T2DV2/classes_GS.csv"
f = open(T2DV2GroundTruth, encoding='latin1', errors='ignore')
gt_CSV = pd.read_csv(f, header=None)
"""
a dictionary that mapping table name and the annotated class label
element in this dictionary: '68779923_0_3859283110041832023': 'Country'
"""
GroundTruth = dict(zip(gt_CSV[0].str.removesuffix(".tar.gz"), gt_CSV[1]))
# print(GroundTruth)
"""
a dictionary that mapping table name and the annotated class label
element in this dictionary: '68779923_0_3859283110041832023': 'Country'
gt_clusters == GroundTruth

ground_t is a dictionary of which key is class label while value is the list
of the table name which belonging to this cluster
like 'GolfPlayer': ['44206774_0_3810538885942465703']


value of gt_clusters
"""
gt_clusters, ground_t, truth = ed.get_concept_files(ed.get_files(samplePath), GroundTruth)
"""
gt_cluster: all the labels of tables  [a,b,c,d,e,...,f]
gt_cluster_dict: table name: index of the cluster label (corresponding to the gt_cluster label index)
{table1: index(a),...,}
"""
gt_cluster = gt_CSV[1].unique()
gt_cluster_dict = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
"""
create lsh indexes of samplePath
"""
indexes = create_or_find_indexes(samplePath)
parameters = dbscan_param_search(samplePath, indexes)
clusters = cluster_discovery(ed.samplePath, parameters)
# for key in gt_clusters:
#   shutil.copy(samplePath + key+".csv", samplePath)
# groundTruthWDCTest = ed.get_concept(ed.WDCsamplePath)
# print(groundTruthWDCTest)
# samplePath = os.getcwd() + "/T2DV2/Test/"
# indexes = create_or_find_indexes(ed.WDCsamplePath)
# parameters = dbscan_param_search(ed.WDCsamplePath, indexes)
# print(parameters)
# clusters = cluster_discovery(ed.WDCsamplePath,parameters)
cluster_dict = {}

for k, v in clusters:
    if cluster_dict.get(v) is None:
        cluster_dict[v] = []
    cluster_dict[v].append(k)

# print(clusters)
print(len(cluster_dict), cluster_dict)

"""
count the most frequent occurring annotated label in the cluster
"""


def most_frequent(list1):
    count = Counter(list1)
    return count.most_common(1)[0][0]


"""
the label of result clusters
"""
clusters_label = {}
"the label index of all tables"
table_label_index = []
"the list of all tables if they have been clustered to the wrong cluster compared to the original label"
false_ones = []
"the original label index of all tables"
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

print(table_label_index, gt_table_label)


# rand_s = metrics.rand_score(table_label_index, gt_table_label)
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
    print(a,b,c,d)
    smc = a  / (a + b + c + d)
    return smc


def metric_Spee(labels_true, labels_pred):
    MI = metrics.mutual_info_score(labels_true, labels_pred)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    rand_score = metrics.rand_score(labels_true, labels_pred)
    adjusted_random_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    FMI = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    smc = SMC(labels_true, labels_pred)
    return {"MI": MI, "NMI": NMI, "AMI": AMI, "random score": rand_score,
            "ARI": adjusted_random_score, "FMI": FMI, 'smc': smc}


print("evaluation metrics is :", metric_Spee(gt_table_label, table_label_index))
# print(clusters_label, false_ones)

df = pd.DataFrame(false_ones, columns=['table name', 'result label', 'true label'])
results = []
for key in clusters_label.keys():
    results.append([key, cluster_dict[key], clusters_label[key]])
df2 = pd.DataFrame(results, columns=['cluster number', 'tables', 'label'])
# baselinePath = os.getcwd()+"/result/baseline/"
baselinePath = os.getcwd() + "/result/subject_column/"
df.to_csv(baselinePath + 'testbeta8.csv', encoding='utf-8', index=False)
df2.to_csv(baselinePath + 'testbeta8_meta.csv', encoding='utf-8', index=False)
"""

    
     
    
"""

'''
dataloader = CSVDataLoader(root_path=(samplePath), encoding='latin-1')
T = ed.get_files(samplePath)
'''
