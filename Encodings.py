from concurrent.futures import ThreadPoolExecutor

from starmie.sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
from argparse import Namespace
import os
from clustering import clustering_results, data_classes, clusteringColumnResults, AgglomerativeClustering_param_search, \
    BIRCH_param_search, cluster_discovery, cluster_Dict, evaluate_cluster
from Utils import mkdir
from SubjectColumnDetection import ColumnType
import TableAnnotation as TA


def extractVectors(dfs, method, dataset, augment, lm, sample, table_order, run_id, check_subject_Column,
                   singleCol=False, SubCol=False, header=False):
    ''' Get model inference on tables
    Args:
        dfs (list of DataFrames): tables to get model inference on
        method (str): model saved path folder
        augment (str): augmentation operator used in vector file path (e.g. 'drop_cell')
        sample (str): sampling method used in vector file path (e.g. 'head')
        table_order (str): 'column' or 'row' ordered
        run_id (int): used in file path
        singleCol (boolean): is this for single column baseline
        SubCol (boolean): is this for subject column baseline
    Return:
        list of features for the dataframe
    '''
    if singleCol:
        model_path = "model/%s/%s/model_%slm_%s_%s_%s_%d_%ssingleCol.pt" % (
            method, dataset, augment, lm, sample, table_order, run_id, check_subject_Column)
    if SubCol:
        model_path = "model/%s/%s/model_%slm_%s_%s_%s_%d_%ssubCol.pt" % (
            method, dataset, augment, lm, sample, table_order, run_id, check_subject_Column)
    if header:
        model_path = "model/%s/%s/model_%slm_%s_%s_%s_%d_%s_header.pt" % (
            method, dataset, augment, lm, sample, table_order, run_id, check_subject_Column)
    else:
        model_path = "model/%s/%s/model_%slm_%s_%s_%s_%d_%s.pt" % (
            method, dataset, augment, lm, sample, table_order, run_id, check_subject_Column)
    print(model_path)
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    # load_checkpoint from sdd/pretain
    model, trainset = load_checkpoint(ckpt)
    print(trainset.tables)
    return inference_on_tables(dfs, model, trainset, batch_size=1024)


def get_df(dataFolder):
    ''' Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
    Return:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    '''
    dataFiles = glob.glob(dataFolder + "/*.csv")
    #print(dataFiles)
    dataDFs = {}
    for file in dataFiles:
        df = pd.read_csv(file) #, lineterminator='\n'
        #print(df.transpose())
        filename = file.split("/")[-1]
        dataDFs[filename] = df
    return dataDFs


def table_features(hp: Namespace):
    DATAFOLDER = "datasets/%s/Test/" % hp.dataset
    tables = get_df(DATAFOLDER)
    print("num dfs:", len(tables))

    dataEmbeds = []
    table_number = len(tables)
    dfs_count = 0
    # Extract model vectors
    cl_features = extractVectors(list(tables.values()), hp.method, hp.dataset, hp.augment_op, hp.lm, hp.sample_meth,
                                 hp.table_order, hp.run_id, hp.check_subject_Column, singleCol=hp.single_column,
                                 SubCol=hp.subject_column, header=hp.header)
    output_path = "result/embedding/%s/vectors/%s/" % (hp.method, hp.dataset)
    mkdir(output_path)
    print(output_path)
    for i, file in enumerate(tables):
        dfs_count += 1
        # get features for this file / dataset
        cl_features_file = np.array(cl_features[i])
        dataEmbeds.append((file, cl_features_file))
        # print(len(tables[file].columns),len(cl_features_file), cl_features_file)

    output_file = "cl_%s_lm_%s_%s_%s_%d_%s.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                   hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.single_column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_singleCol.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                                 hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.subject_column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_subCol.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.header:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_header.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)


    output_path += output_file
    if hp.save_model:
        pickle.dump(dataEmbeds, open(output_path, "wb"))


def hierarchical_clustering(hp: Namespace):
    print("hierarchical_clustering ...")
    datafile_path = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", hp.dataset)
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    gt_filename = "groundTruth.csv"
    ground_truth = os.path.join(os.getcwd(), "datasets", hp.dataset, gt_filename)
    gt = pd.read_csv(ground_truth, encoding='latin1')
    gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, ground_truth)
    # needs to delete in the future when all labeling is done
    colname = 'superclass' if hp.dataset == "TabFact" else "Label"
    tables_with_class = gt[gt[colname].notnull()]
    clusters = tables_with_class[colname].unique()
    files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn]

    for file in files[1:]:
        store_path = os.path.join(os.getcwd(), "result/starmie", hp.dataset, "clusteringModel", file[:-4])
        mkdir(store_path)
        dict_file = {}
        F = open(os.path.join(datafile_path, file), 'rb')
        content = pickle.load(F)

        Z = []
        T = []
        for vectors in content:
            T.append(vectors[0][:-4])

            vec_table = np.mean(vectors[1], axis=0)
            Z.append(vec_table)
        Z = np.array(Z)
        ta = "result/starmie/%s/clusteringModel" % hp.dataset
        dp = os.path.join(os.getcwd(), ta)
        for folder in os.listdir(dp):
            cluster = "Agglomerative_clustering_results.pickle"
            F_cluster = open(os.path.join(dp, folder, cluster), 'rb')
            content_cluster = pickle.load(F_cluster)
            methods_metrics = {}
            metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
            clusters = cluster_discovery(content_cluster, T)
            cluster_dict = cluster_Dict(clusters)
            table_dict = None
            table_dict = {T[i]: Z[i] for i in range(0, len(T))}
            # print("cluster_dict",cluster_dict)
            metric_dict = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, tables_gt=table_dict)
            print("cluster_dict", cluster_dict)
            metric_df = pd.DataFrame([metric_dict])
            metric_value_df = pd.concat([metric_value_df, metric_df])
            dict_file["Agglomerative"] = cluster_dict

            mean_metric = metric_value_df.mean()
            methods_metrics["Agglomerative"] = mean_metric

            e_df = pd.DataFrame()
            for i, v in methods_metrics.items():
                e_df = pd.concat([e_df, v.rename(i)], axis=1)

            store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
            store_path += "All/"
            mkdir(store_path)
            e_df.to_csv(store_path + file[:-4] + '_metrics.csv', encoding='utf-8')

        """try:
                print(hp.clustering)
                parameters=[]
                if hp.clustering == "Agglomerative":
                    parameters = AgglomerativeClustering_param_search(Z, len(clusters))
                if hp.clustering == "birch":
                    parameters = BIRCH_param_search(Z, len(clusters))
                pickle_file_path = "%s_clustering_results.pickle" % hp.clustering
                store_name = os.path.join(store_path, pickle_file_path)
                with open(store_name, 'wb') as fp:
                    pickle.dump(parameters, fp)
        except ValueError as e:
            print(e)
            continue"""


def silm_clustering(hp: Namespace):
    dicts = {}
    files = []
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"

    # subject_path = os.getcwd() + "/datasets/" + hp.dataset + "/SubjectColumn/"
    if hp.method == "starmie":
        files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn and 'roberta' in fn]  # pkl
    if hp.subjectCol:
        F_cluster = open(os.path.join(os.getcwd(), "datasets/" + hp.dataset, "SubjectCol.pickle"), 'rb')
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
                                 "BIRCH"]  # "DBSCAN", "GMM", "KMeans", "OPTICS",Agglomerative  BIRCH , "BIRCH"
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


"""
 
 #        + "_" + str(hp.sample_meth) + "_" + str(hp.table_order) + '_' + str(hp.run_id) + "singleCol.pt"
        model_path = "model/%s/model_%s_%s_%s_%dsingleCol.pt" % (  sample, table_order, run_id)

# Extract model vectors, and measure model inference time
start_time = time.time()
cl_features = extractVectors(list(dfs.values()), dataFolder, ao, sm, table_order, run_id, singleCol=isSingleCol)
inference_times += time.time() - start_time
print("%s %s inference time: %d seconds" % (dataFolder, dir, time.time() - start_time))
for i, file in enumerate(dfs):
    dfs_count += 1
    # get features for this file / dataset
    cl_features_file = np.array(cl_features[i])
    dataEmbeds.append((file, cl_features_file))
if dir == 'santos-query':
    saveDir = 'query'
elif dir == 'benchmark':
    saveDir = 'datalake'
else:
    saveDir = dir

if isSingleCol:
    output_path = "data/%s/vectors/cl_%s_%s_%s_%s_%d_singleCol.pkl" % (dataFolder, saveDir, ao, sm, table_order, run_id)
else:
    output_path = "data/%s/vectors/cl_%s_%s_%s_%s_%d.pkl" % (dataFolder, saveDir, ao, sm, table_order, run_id)
if hp.save_model:
    pickle.dump(dataEmbeds, open(output_path, "wb"))
print("Benchmark: ", dataFolder)
print("--- Total Inference Time: %s seconds ---" % (inference_times))


"""


def column_gts_WDC(dataset):
    """gt_clusters: tablename.Columnname:corresponding label dictionary. e.g.{'SOTAB_0.a1': 'Game', 'SOTAB_0.b1': 'date',...}
            ground_t: label and its tables. e.g. {'Game': ['SOTAB_0.a1'], 'Newspaper': ['SOTAB_0.b1', 'SOTAB_0.c1'],...}
            gt_cluster_dict: dictionary of index: label
            like {'Game': 0, 'date': 1, ...}
    """
    groundTruth_file = os.getcwd() + "/datasets/" + dataset + "/column_gtf.xlsx"
    ground_truth_df = pd.read_excel(groundTruth_file, sheet_name=0)
    Superclass = ground_truth_df['superclass'].unique()
    column_cluster = {}
    taleBasicClass = {}
    for table in Superclass:
        grouped_df = ground_truth_df[ground_truth_df['Superclass'] == table]
        grouped_classes = list(set(grouped_df['Table_Cluster_Label'].tolist()))
        if len(grouped_classes) > 1:
            taleBasicClass[table] = grouped_classes
        unique_column_cluster = grouped_df['Column_super_label'].unique()
        column_cluster[table] = {}
        for cluster in unique_column_cluster:
            # column_cluster[table][cluster] = []
            Cluster_col = []
            filtered_rows = grouped_df[grouped_df['Column_super_label'] == cluster]
            for index, row in filtered_rows.iterrows():
                Cluster_col.append(f"{row['Source_Dataset']}.{row['column1']}")
                Cluster_col.append(f"{row['Target_Dataset']}.{row['column2']}")
            column_cluster[table][cluster] = list(set(Cluster_col))
    return column_cluster, taleBasicClass


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


"""
print(column_gts("WDC")[1])
for i in column_gts("WDC")[0].items():
    print(i)
"""


def columncluster_gt(tablegt, column_cluster: dict):
    if tablegt in column_cluster.keys():
        clusters = column_cluster[tablegt]
        ground_t = clusters
        gt_clusters = {}
        ck = list(clusters.keys())
        gt_cluster_dict = {ck[i]: i for i in range(0, len(ck))}
        for cluster, values in clusters.items():
            for value in values:
                gt_clusters[value] = cluster

        return gt_clusters, ground_t, gt_cluster_dict
    else:
        print(tablegt)
        return 0, 0, 0


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
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(colCluster, index, clu, content, Ground_t, Zs, Ts, data_path, hp, embedding_file,
                                   gt_clusters, gt_cluster_dict, ground_t) for index, clu in
                   enumerate(list(gt_cluster_dict.keys()))]

        # wait all parallel task to complete
        for future in futures:
            future.result()

    print("All parallel tasks completed.")


def colCluster(index, clu, content, Ground_t, Zs, Ts, data_path, hp, embedding_file, gt_clusters, gt_cluster_dict,
               ground_t):
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
    print( f"columns NO :{len(Zs[clu])}, cluster NO: {len(gt_cluster_dict[clu])}"
            f" \n ground truth class {clu} ")
    Zs[clu] = np.array(Zs[clu])
    try:
        clustering_method = ["BIRCH", "Agglomerative"]  # "DBSCAN", "GMM", "KMeans", "OPTICS",
        methods_metrics = {}
        store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
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
        with open(store_path + str(index) + '_colcluster_dict.pickle', 'wb') as handle:
            pickle.dump(clusters_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except ValueError as e:
        print(e)


def starmie_clusterHierarchy(hp: Namespace):
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    ground_truth_table = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    Gt_clusters, Gt_cluster_dict = data_classes(data_path, ground_truth_table)[0], \
                                   data_classes(data_path, ground_truth_table)[2]
    print("Gt_clusters,Gt_cluster_dict", Gt_clusters, Gt_cluster_dict)
    tables = []
    for key, value in Gt_cluster_dict.items():
        if value in ['People', "Animal"]:
            tables.append(value)


def files_columns_running(hp: Namespace):
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if fn.endswith(hp.embedMethod+'.pkl') and hp.embed in fn]
    starmie_columnClustering(files[0], hp)
