import math
import os
import pickle
from argparse import Namespace
import time

import pandas as pd

from ClusterHierarchy.ClusterDecompose import tree_consistency_metric
from ClusterHierarchy.JaccardMetric import JaccardMatrix
from EndToEnd.EndToEnd import read_col_Embeddings, find_cluster_embeddings, type_info, endToEndRelationship
from TableCluster.tableClustering import read_embeddings_P1
from Utils import mkdir
from clustering import clustering_results, clustering

def get_files_size(csv_files, folder_path):
    total_size = 0
    for file in csv_files:
        file_path = os.path.join(folder_path, file+".csv")
        file_size = os.path.getsize(file_path)
        total_size+=file_size
    return total_size/1024

def Running(hp: Namespace):
    F = open(f"datasets/{hp.dataset}/SubjectCol.pickle", 'rb')
    SE = pickle.load(F)
    Z, T = read_embeddings_P1(hp.P1Embed, hp.subjectCol, hp.dataset, SelectedNumber=hp.tableNumber)
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    size = get_files_size(T, data_path)
    clustering_method = hp.clustering
    # print("numE", hp.estimateNumber)
    print("size of tables: ",size)
    cluster_dict, metric_dict = clustering_results(Z, T, data_path, clustering_method, numEstimate=hp.estimateNumber)
    P1_time = metric_dict["Clustering time"]
    cluster_dict_all = type_info(cluster_dict, SE, hp.dataset, noLabel=True)

    filepath = f"datasets/{hp.dataset}/Test/"
    content = read_col_Embeddings(hp.P23Embed, hp.dataset, selected_tables=T)
    store_path = f"result/Runtime/{hp.dataset}/"
    mkdir(store_path)

    P2_time = 0
    P3_time = 0

    for name in cluster_dict.keys():
        cluster = cluster_dict[name]

        input_data, names = find_cluster_embeddings(cluster, content, SE, filepath)
        MIN = math.ceil(len(input_data) / 40) if math.ceil(len(input_data) / 40) > 2 else 2
        # print("index cluster and # of its attributes", name, len(input_data))
        start_time_P2 = time.time()
        colcluster_dict = clustering(input_data, names, MIN, clustering_method,
                                     max=2 * MIN)
        end_time_P2 = time.time()
        P2_time_cluster = end_time_P2 - start_time_P2
        P2_time += P2_time_cluster
        colCluster = {index: {'name': None, 'cluster': cluster} for index, cluster in colcluster_dict.items()}
        cluster_dict_all[name]["attributes"] = colCluster

        start_time_P3 = time.time()
        jaccard_score = JaccardMatrix(colcluster_dict, data_path)[2]
        TCS, ALL_path, simple_tree = tree_consistency_metric(cluster, jaccard_score, hp.P23Embed,
                                                             hp.dataset, sliceInterval=hp.intervalSlice,
                                                             delta=hp.delta,
                                                             store_results=False, Test=False)
        end_time_P3 = time.time()
        P3_time_cluster = end_time_P3 - start_time_P3
        P3_time += P3_time_cluster

    start_time_P4 = time.time()
    endToEndRelationship(hp, cluster_dict_all)
    end_time_P4 = time.time()
    P4_time = end_time_P4 - start_time_P4

    if "runningTime.csv" not in os.listdir(store_path):
        data = {
            'tableNumber': [hp.tableNumber],
            'size':[size],
            'P1': [P1_time],
            'P2': [P2_time],
            'P3': [P3_time],
            'P4': [P4_time]
        }
        df = pd.DataFrame(data, index=[1])

    else:
        df = pd.read_csv(os.path.join(store_path, "runningTime.csv"))
        new_data = {'tableNumber': hp.tableNumber, 'size': size, 'P1': P1_time, 'P2': P2_time, 'P3': P3_time, 'P4': P4_time}
        new_row = pd.DataFrame(new_data, index=[len(df) + 1])
        df = pd.concat([df, new_row])
    df.to_csv(os.path.join(store_path, "runningTime.csv"))
