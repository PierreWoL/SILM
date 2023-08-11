import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
from argparse import Namespace
import os
from clustering import clustering_results, data_classes, clusteringColumnResults
from Utils import mkdir

from ClusterHierarchy.ClusterDecompose import hierarchicalColCluster


def silm_clustering(hp: Namespace):
    dicts = {}
    files = []
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"

    if hp.method == "starmie":
        files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn and 'roberta' in fn]  # pkl
    if hp.subjectCol:
        F_cluster = open(os.path.join(os.getcwd(),
                                      "datasets/" + hp.dataset, "SubjectCol.pickle"), 'rb')
        SE = pickle.load(F_cluster)
    else:
        SE = {}
    ground_truth = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"

    available_data = pd.read_csv(ground_truth)["fileName"].unique().tolist()
    for file in files[2:]:
        print(file, hp.subjectCol)
        dict_file = {}
        F = open(datafile_path + file, 'rb')
        content = pickle.load(F)
        Z = []
        T = []
        content = [vectors for vectors in content if vectors[0] in available_data]
        print(len(content))
        for vectors in content:
            T.append(vectors[0][:-4])
            if hp.subjectCol:
                NE_list, headers, types = SE[vectors[0]]
                if NE_list:
                    vec_table = vectors[1][NE_list[0]]
                else:
                    vec_table = np.mean(vectors[1], axis=0)
            else:
                vec_table = np.mean(vectors[1], axis=0)
            Z.append(vec_table)
        """has_nan = np.isnan(Z).any()
        if has_nan:
            print(vectors[0],vectors[1], np.isnan(vec_table).any(),vec_table)"""
        Z = np.array(Z)
        try:
            clustering_method = ["Agglomerative",
                                 "BIRCH"]
            methods_metrics = {}
            for method in clustering_method:
                print(method)
                metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
                for i in range(0, 1):
                    cluster_dict, metric_dict = clustering_results(Z, T, data_path, ground_truth, method)

                    metric_df = pd.DataFrame([metric_dict])
                    metric_value_df = pd.concat([metric_value_df, metric_df])
                    dict_file[method + "_" + str(i)] = cluster_dict

                mean_metric = metric_value_df.mean()
                methods_metrics[method] = mean_metric

            e_df = pd.DataFrame()
            for i, v in methods_metrics.items():
                e_df = pd.concat([e_df, v.rename(i)], axis=1)

            store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
            if hp.subjectCol is True:
                store_path += "Subject_Col/"
            else:
                store_path += "All/"
            mkdir(store_path)
            e_df.to_csv(store_path + file[:-4] + '_metrics.csv', encoding='utf-8')
            print(e_df)
            dicts[file] = dict_file
        except ValueError as e:
            print(e)
            continue
        break
    with open(datafile_path + 'cluster_dict.pickle', 'wb') as handle:
        pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def column_gts(dataset):
    """
     gt_clusters: tablename.colIndex:corresponding label dictionary.
      e.g.{Topclass1: {'T1.a1': 'ColLabel1', 'T1.b1': 'ColLabel2',...}}
    ground_t: label and its tables. e.g.
     {Topclass1:{'ColLabel1': ['T1.a1'], 'ColLabel2': ['T1.b1'', 'T1.c1'],...}}
    gt_cluster_dict: dictionary of index: label
    like  {Topclass1:{'ColLabel1': 0, 'ColLabel2': 1, ...}}
    """
    groundTruth_file = os.getcwd() + "/datasets/" + dataset + "/column_gt.csv"
    ground_truth_df = pd.read_csv(groundTruth_file, encoding='latin1')
    print(len(ground_truth_df['ColumnLabel'].unique()))
    Superclass = ground_truth_df['TopClass'].unique()
    gt_clusters = {}
    ground_t = {}
    gt_cluster_dict = {}
    for classTable in Superclass:
        gt_clusters[classTable] = {}
        ground_t[classTable] = {}
        grouped_df = ground_truth_df[ground_truth_df['TopClass'] == classTable]

        for index, row in grouped_df.iterrows():
            gt_clusters[classTable][row["fileName"][0:-4] + "." + str(row["colName"])] = str(row["ColumnLabel"])
            if row["ColumnLabel"] not in ground_t[classTable].keys():
                ground_t[classTable][str(row["ColumnLabel"])] = [row["fileName"][0:-4] + "." + str(row["colName"])]
            else:
                ground_t[classTable][str(row["ColumnLabel"])].append(row["fileName"][0:-4] + "." + str(row["colName"]))
        gt_cluster = pd.Series(gt_clusters[classTable].values()).unique()
        gt_cluster_dict[classTable] = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
        # print(len(gt_cluster_dict[classTable]))
    print(len(gt_clusters))
    return gt_clusters, ground_t, gt_cluster_dict


def starmie_columnClustering(embedding_file: str, hp: Namespace):
    print(embedding_file)
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    target_path = os.getcwd() + "/result/Valerie/Column/" + hp.dataset + "/"
    mkdir(target_path)
    ground_truth_table = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table)
    F = open(datafile_path + embedding_file, 'rb')
    content = pickle.load(F)

    # content is the embeddings for all datasets
    Zs = {}
    Ts = {}
    gt_clusters, ground_t, gt_cluster_dict = column_gts(hp.dataset)
    with open(os.path.join(target_path, '_gt_cluster.pickle'),
              'wb') as handle:
        pickle.dump(list(gt_cluster_dict.keys()), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(colCluster, index, clu, content, Ground_t, Zs, Ts, data_path, hp, embedding_file,
                                   gt_clusters, gt_cluster_dict) for index, clu in
                   enumerate(list(gt_cluster_dict.keys()))]

        # wait all parallel task to complete
        for future in futures:
            future.result()

    print("All parallel tasks completed.")


def colCluster(index, clu, content, Ground_t, Zs, Ts, data_path, hp, embedding_file, gt_clusters, gt_cluster_dict):
    clusters_result = {}
    tables_vectors = [vector for vector in content if vector[0].removesuffix(".csv") in Ground_t[clu]]

    Ts[clu] = []
    Zs[clu] = []
    for vector in content:
        if vector[0].removesuffix(".csv") in Ground_t[clu]:
            table = pd.read_csv(data_path + vector[0], encoding="latin1")
            for i in range(0, len(table.columns)):
                Ts[clu].append(f"{vector[0][0:-4]}.{table.columns[i]}")
                Zs[clu].append(vector[1][i])


    
    Zs[clu] = np.array(Zs[clu])
    store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
    pickle_name = store_path + str(index) + '_colcluster_dict.pickle'
    clustering_method = ["Agglomerative"]

    if len(Zs[clu])<2500:
        print(f"index: {index} columns NO :{len(Zs[clu])}, cluster NO: {len(gt_cluster_dict[clu])}"
          f" \n ground truth class {clu} ")

        try:
            methods_metrics = {}
            embedding_file_path = embedding_file.split(".")[0]
            col_example_path = os.path.join(store_path, "example", embedding_file_path)
            store_path += "All/" + embedding_file_path + "/column/"
            mkdir(store_path)
            mkdir(col_example_path)
            for method in clustering_method:

                metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
                for i in range(0, 1):
                    cluster_dict, metric_dict = clusteringColumnResults(Zs[clu], Ts[clu], gt_clusters[clu],
                                                                        gt_cluster_dict[clu], method,
                                                                        folderName=col_example_path,
                                                                        filename=f"{str(index)}.{method}")
                    if i == 0:
                        clusters_result[method] = cluster_dict
                    metric_df = pd.DataFrame([metric_dict])
                    metric_value_df = pd.concat([metric_value_df, metric_df])
                mean_metric = metric_value_df.mean()
                methods_metrics[method] = mean_metric
            print("methods_metrics is", methods_metrics)

            e_df = pd.DataFrame()
            for i, v in methods_metrics.items():
                e_df = pd.concat([e_df, v.rename(i)], axis=1)
            e_df.to_csv(store_path + str(index) + '_ColumnMetrics.csv', encoding='utf-8')
            with open(pickle_name, 'wb') as handle:
                pickle.dump(clusters_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for meth in clustering_method:
                hierarchicalColCluster(meth, str(index) + '_colcluster_dict.pickle', hp)
        except ValueError as e:
            print(e)


def starmie_clusterHierarchy(hp: Namespace):
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    ground_truth_table = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    Gt_clusters, Gt_cluster_dict = data_classes(data_path, ground_truth_table)[0], \
                                   data_classes(data_path, ground_truth_table)[2]
    # print("Gt_clusters,Gt_cluster_dict", Gt_clusters, Gt_cluster_dict)
    tables = []
    for key, value in Gt_cluster_dict.items():
        if value in ['People', "Animal"]:
            tables.append(value)

def files_columns_running(hp: Namespace):
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if fn.endswith("_" + hp.embedMethod + '.pkl') and hp.embed in fn]
    starmie_columnClustering(files[0], hp)
