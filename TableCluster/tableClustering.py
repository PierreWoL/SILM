import time

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
from argparse import Namespace
import os
from clustering import clustering_results, data_classes, clusteringColumnResults, inputData
from Utils import mkdir

from ClusterHierarchy.ClusterDecompose import hierarchicalColCluster

import sys


# sys.setrecursionlimit(3000)  # or another higher value

def silm_clustering(hp: Namespace):
    dicts = {}
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"  # /Subject attribute/None
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"

    # Read the groung truth hierarchy
    # F_graph = open(os.path.join(os.getcwd(),  "datasets/" + hp.dataset, "graphGroundTruth.pkl"), 'rb')
    # graph_gt = pickle.load(F_graph)

    files = [fn for fn in os.listdir(datafile_path) if
             '.pkl' in fn and f"_{hp.embed}_" in fn and 'subCol' in fn]  # pkl  and 'cell' in fn and 'subCol' in fn Pretrain
    if hp.subjectCol:
        F_cluster = open(os.path.join(os.getcwd(),
                                      "datasets/" + hp.dataset, "SubjectCol.pickle"), 'rb')
        SE = pickle.load(F_cluster)
    else:
        SE = {}
    ground_truth = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    available_data = pd.read_csv(ground_truth)["fileName"].unique().tolist()

    store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
    if hp.subjectCol is True:
        store_path += "Subject_Col/"
    else:
        store_path += "All/"
    mkdir(store_path)

    for file in files:
        print(file, hp.subjectCol)
        dict_file = {}
        F = open(datafile_path + file, 'rb')
        content = pickle.load(F)
        Z = []
        T = []
        content = [vectors for vectors in content if vectors[0] in available_data]

        # for showing the first item in content
        # print(len(content), content[0])
        for vectors in content:
            T.append(vectors[0][:-4])
            table = pd.read_csv(data_path + vectors[0], encoding="latin1")
            if hp.subjectCol:
                if file.endswith("subCol.pkl"):
                    vec_table = np.mean(vectors[1], axis=0)
                else:
                    NE_list, headers, types = SE[vectors[0]]
                    if NE_list:
                        vec_table = vectors[1][NE_list[0]]
                    else:
                        vec_table = np.mean(vectors[1], axis=0)
            else:
                vec_table = np.mean(vectors[1], axis=0)
            Z.append(vec_table)
        #
        """has_nan = np.isnan(Z).any()
        if has_nan:
            print(vectors[0],vectors[1], np.isnan(vec_table).any(),vec_table)"""
        Z = np.array(Z)
        try:

            clustering_method = ["Agglomerative"]  # , "BIRCH"
            methods_metrics = {}
            for method in clustering_method:
                # print(method)
                metric_value_df = pd.DataFrame(
                    columns=["Random Index", "Purity", "Clustering time",
                             "Evaluation time"])  # "ARI", "MI", "NMI", "AMI" "Average cluster consistency score"
                for i in range(0, 1):
                    new_path = os.path.join(store_path, file[:-4])
                    mkdir(new_path)
                    cluster_dict, metric_dict = clustering_results(Z, T, data_path, ground_truth, method,
                                                                   folderName=new_path)  # , graph=graph_gt
                    # print(cluster_dict)
                    metric_df = pd.DataFrame([metric_dict])
                    metric_value_df = pd.concat([metric_value_df, metric_df])
                    dict_file[method + "_" + str(i)] = cluster_dict
                mean_metric = metric_value_df.mean()
                methods_metrics[method] = mean_metric
            e_df = pd.DataFrame()
            for i, v in methods_metrics.items():
                e_df = pd.concat([e_df, v.rename(i)], axis=1)
            e_df.to_csv(store_path + file[:-4] + '_metrics.csv', encoding='utf-8')
            print(e_df)
            dicts[file] = dict_file

        except ValueError as e:
            print(e)
            continue


def baselineTypeClustering(hp: Namespace):
    dicts = {}
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    isTabFact, subCol = False, False

    store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
    if hp.subjectCol is True:
        subCol = True
        store_path += "Subject_Col/"
    else:
        store_path += "All/"
    ground_truth = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    dict_file = {}
    nameFile = os.getcwd() + "/datasets/" + hp.dataset + "/naming.csv"
    Z, T = inputData(data_path, 0.6, 5, hp.embed, subjectCol=subCol)
    print(Z, T)
    try:

        clustering_method = ["Agglomerative"]  # , "BIRCH"
        methods_metrics = {}
        for method in clustering_method:
            print(method)
            metric_value_df = pd.DataFrame(
                columns=["Random Index", "Purity"])
            for i in range(0, 1):
                new_path = os.path.join(store_path, "D3L")
                mkdir(new_path)
                cluster_dict, metric_dict = clustering_results(Z, T, data_path, ground_truth,
                                                               method)  # , graph=graph_gtv
                # print(cluster_dict)
                metric_df = pd.DataFrame([metric_dict])
                metric_value_df = pd.concat([metric_value_df, metric_df])
                dict_file[method + "_" + str(i)] = cluster_dict
            mean_metric = metric_value_df.mean()
            methods_metrics[method] = mean_metric
        e_df = pd.DataFrame()
        for i, v in methods_metrics.items():
            e_df = pd.concat([e_df, v.rename(i)], axis=1)
        e_df.to_csv(store_path + f"D3L_{hp.embed}_metrics.csv", encoding='utf-8') if subCol is False \
            else e_df.to_csv(store_path + f"D3LSubCol_{hp.embed}_metrics.csv", encoding='utf-8')
        print(e_df)
        dicts["D3L"] = dict_file
    except ValueError as e:
        print(e)


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
    Superclass = ground_truth_df['TopClass'].dropna().unique()

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


def starmie_columnClustering(hp: Namespace, embedding_file: str = None):
    print(embedding_file)
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    target_path = os.getcwd() + "/result/SILM/Column/" + hp.dataset + "/"
    mkdir(target_path)
    ground_truth_table = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table, Nochange=True)
    gt_clusters, ground_t, gt_cluster_dict = column_gts(hp.dataset)
    clustering_method = ["Agglomerative"]  #
    with open(os.path.join(target_path, '_gt_cluster.pickle'),
              'wb') as handle:
        pickle.dump(list(gt_cluster_dict.keys()), handle, protocol=pickle.HIGHEST_PROTOCOL)

    if embedding_file is None:
        Z, T = inputData(data_path, 0.6, 5, hp.embed, column=True)
        content = []
        for index, col in enumerate(Z.columns):
            content.append((T[index], np.array(Z[col])))
        embedding_file = f"D3L_{hp.embed}"
    else:
        F = open(datafile_path + embedding_file, 'rb')
        content = pickle.load(F)
        # content is the embeddings for all datasets
    Zs = {}
    Ts = {}
    for index, clu in enumerate(list(gt_cluster_dict.keys())):
        startTimeCC = time.time()
        colCluster(clustering_method, index, clu, content, Ground_t, Zs, Ts, data_path, hp, embedding_file,
                   gt_clusters,
                   gt_cluster_dict)
        checkfile = f"{index}_colcluster_dict.pickle"
        datafile_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                     "All/" + embedding_file[:-4] + "/column")
        endTimeCC = time.time()
        TimespanCC = endTimeCC - startTimeCC

        if os.path.isfile(os.path.join(datafile_path, checkfile)):  # 816 1328
            startTimeTH = time.time()
            ClusterDecompose(clustering_method, index, embedding_file, Ground_t, hp)
            endTimeTH = time.time()
            TimespanTH = endTimeTH - startTimeTH
            timing = pd.DataFrame({'Column clustering': TimespanCC, 'Hierarchy Inference ': TimespanTH},
                                  columns=['type', 'time'])
            timing.to_csv(os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                       "All/" + embedding_file[:-4], "timing.csv"))

    # with ThreadPoolExecutor(max_workers=3) as executor:
    # futures = [executor.submit(colCluster, clustering_method, index, clu, content, Ground_t, Zs, Ts, data_path, hp,
    # embedding_file,
    # gt_clusters, gt_cluster_dict) for index, clu in
    # enumerate(list(gt_cluster_dict.keys()))]
    # wait all parallel task to complete
    # for future in futures:
    # future.result()
    # for index, clu in enumerate(list(gt_cluster_dict.keys())):
    # ClusterDecompose(clustering_method, index, embedding_file, Ground_t, hp)
    # print("All parallel tasks completed.")


def inferenceHierarchy(embedding_file: str, hp: Namespace):
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    target_path = os.getcwd() + "/result/SILM/Column/" + hp.dataset + "/"
    mkdir(target_path)
    ground_truth_table = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table, Nochange=True)
    gt_clusters, ground_t, gt_cluster_dict = column_gts(hp.dataset)
    with open(os.path.join(target_path, '_gt_cluster.pickle'),
              'wb') as handle:
        pickle.dump(list(gt_cluster_dict.keys()), handle, protocol=pickle.HIGHEST_PROTOCOL)
    clustering_method = ["Agglomerative"]  #

    for index, clu in enumerate(list(gt_cluster_dict.keys())):
        checkfile = f"{index}_colcluster_dict.pickle" if hp.phaseTest is False else f"{index}_colcluster_dictGT.pickle"
        datafile_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                     "All/" + embedding_file[:-4] + "/column")
        print(os.path.join(datafile_path, checkfile))
        if os.path.isfile(os.path.join(datafile_path, checkfile)):  # 816 1328

            if hp.phaseTest is True:
                clustering_method = ["groundTruth"]
                store_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                          "All/" + embedding_file[0:-4] + f"/column/")
                print(store_path)
                mkdir(store_path)
                path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                    "All/" + embedding_file[0:-4] + f"/column/{str(index)}_colcluster_dictGT.pickle")

                if os.path.exists(path) is False:
                    dict_groundTruth = {'groundTruth': ground_t[clu]}
                    # print(path)
                    with open(path, 'wb') as handle:
                        pickle.dump(dict_groundTruth, handle, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                ClusterDecompose(clustering_method, index, embedding_file, Ground_t, hp)
            except:
                print("error", index, clu)


def colCluster(clustering_method, index, clu, content, Ground_t, Zs, Ts, data_path, hp, embedding_file, gt_clusters,
               gt_cluster_dict):
    clusters_result = {}
    # tables_vectors = [vector for vector in content if vector[0].removesuffix(".csv") in Ground_t[clu]]
    Ts[clu] = []
    Zs[clu] = []
    for vector in content:
        if embedding_file.endswith("_column.pkl"):
            if vector[0] in gt_clusters[clu].keys():
                Ts[clu].append(vector[0])
                Zs[clu].append(vector[1][0])
        elif "D3L" in embedding_file:

            if vector[0] in gt_clusters[clu].keys():
                Ts[clu].append(vector[0])
                Zs[clu].append(vector[1])
        else:
            if vector[0].removesuffix(".csv") in Ground_t[clu]:
                table = pd.read_csv(data_path + vector[0], encoding="latin1")
                for i in range(0, len(table.columns)):
                    Ts[clu].append(f"{vector[0][0:-4]}.{table.columns[i]}")
                    Zs[clu].append(vector[1][i])
    Zs[clu] = np.array(Zs[clu]).astype(np.float32)
    store_path = os.getcwd() + "/result/SILM/" + hp.dataset + "/"
    embedding_file_path = embedding_file.split(".")[0]
    store_path += "All/" + embedding_file_path + "/column/"
    mkdir(store_path)
    # if len(Zs[clu]) > 20000:  # 816 1328 5000<len(Zs[clu]) <
    if os.path.isfile(store_path + str(index) + '_ColumnMetrics.csv'):
        print(f"exists! index: {index} columns NO :{len(Zs[clu])}, cluster NO: {len(gt_cluster_dict[clu])} ")
    else:
        print(f"index: {index} columns NO :{len(Zs[clu])}, cluster NO: {len(gt_cluster_dict[clu])}"
              f" \n ground truth class {clu} {Zs[clu].dtype}")

        methods_metrics = {}

        col_example_path = os.path.join(store_path, "example", embedding_file_path)

        pickle_name = store_path + str(index) + '_colcluster_dict.pickle'

        mkdir(col_example_path)
        for method in clustering_method:
            metric_value_df = pd.DataFrame(
                columns=["MI", "NMI", "AMI", "random Index", "ARI", "FMI", "purity", "Clustering time",
                         "Evaluation time"])
            for i in range(0, 1):
                # TODO: add the naming part and ground truth of attribute name part
                cluster_dict, metric_dict = clusteringColumnResults(Zs[clu], Ts[clu], gt_clusters[clu],
                                                                    gt_cluster_dict[clu], method,
                                                                    folderName=col_example_path,
                                                                    filename=f"{str(index)}.{method}")
                # print(cluster_dict)
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


def ClusterDecompose(clustering_method, index, embedding_file, Ground_t, hp):
    if hp.phaseTest is True:
        filename = str(index) + '_colcluster_dictGT.pickle'
    else:
        filename = str(index) + '_colcluster_dict.pickle'
    print(filename)
    for meth in clustering_method:
        # try:
        hierarchicalColCluster(meth, filename, embedding_file[0:-4], Ground_t,
                               hp)

    # except:
    # continue


def files_columns_running(hp: Namespace):
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"

    if hp.baseline is True:
        starmie_columnClustering(hp)
    files = [fn for fn in os.listdir(datafile_path) if
             '.pkl' in fn and f"_{hp.embed}_" in fn]  # if fn.endswith('_column.pkl') and hp.embed in fn] and 'Pretrain' in fn and 'subCol' not in fn
    files = [fn for fn in files if fn.endswith("_column.pkl") and '8' in fn]
    print(len(files), files)
    for file in files[hp.slice_start:hp.slice_stop]:  # [hp.slice_start:hp.slice_stop]
        starmie_columnClustering(hp, file)


def files_hierarchyInference(hp: Namespace):
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if
             fn.endswith('.pkl') and f"_{hp.embed}_" in fn]  # and 'SCT6' in fn and 'header' not in fn
    files = [fn for fn in files if not fn.endswith("_column.pkl") and 'Pretrain' not in fn][hp.slice_start:hp.slice_stop]

    print(files, len(files))
    for file in files:  # [hp.slice_start:hp.slice_stop]:
        # try:
        inferenceHierarchy(file, hp)
    # except:
    # continue
