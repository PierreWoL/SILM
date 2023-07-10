from starmie.sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
from argparse import Namespace
import os
from clustering import clustering_results, data_classes, clusteringColumnResults, AgglomerativeClustering_param_search, \
    BIRCH_param_search
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
    print(dataFiles)
    dataDFs = {}
    for file in dataFiles:
        df = pd.read_csv(file, lineterminator='\n')
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

    if hp.single_column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_singleCol.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                                 hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.subject_column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_subCol.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.header:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_header.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)
    else:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s.pkl" % (hp.augment_op, hp.lm, hp.sample_meth,
                                                       hp.table_order, hp.run_id, hp.check_subject_Column)

    output_path += output_file
    if hp.save_model:
        pickle.dump(dataEmbeds, open(output_path, "wb"))


def hierarchical_clustering(hp: Namespace):
    print("hierarchical_clustering ...")
    datafile_path = os.path.join(os.getcwd(), "result/embedding/starmie/vectors", hp.dataset)
    gt_filename = "01SourceTables.csv" if hp.dataset == "TabFact" else "groundTruth.csv"
    ground_truth = os.path.join(os.getcwd(), "datasets",hp.dataset, gt_filename)
    gt = pd.read_csv(ground_truth, encoding='latin1')

    # needs to delete in the future when all labeling is done
    colname = 'superclass' if hp.dataset == "TabFact" else "Label"
    tables_with_class = gt[gt[colname].notnull()]
    clusters = tables_with_class[colname].unique()
    files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn]
    
    for file in files:
        store_path = os.path.join(os.getcwd(), "result/starmie", hp.dataset, "clusteringModel",file[:-4])
        mkdir(store_path)
        print(file)
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
        print(Z,T)
        try:
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
            continue


def starmie_clustering_old(hp: Namespace):
    dicts = {}
    files = []
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"

    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"
    subject_path = os.getcwd() + "/datasets/" + hp.dataset + "/SubjectColumn/"
    if hp.method == "starmie":
        files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn]
    ground_truth = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    for file in files:
        dict_file = {}
        F = open(datafile_path + file, 'rb')
        content = pickle.load(F)
        Z = []
        T = []
        for vectors in content:
            T.append(vectors[0][:-4])
            if hp.is_sub is True:
                table = pd.read_csv(data_path + vectors[0])
                Sub_cols_header = []
                if vectors[0] in [fn for fn in os.listdir(subject_path) if
                                  '.csv' in fn and 'Metadata' not in fn]:
                    Sub_cols = pd.read_csv(subject_path + vectors[0])
                    for column in Sub_cols.columns.tolist():
                        if column in table.columns.tolist():
                            Sub_cols_header.append(table.columns.tolist().index(column))

                else:
                    anno = TA.TableColumnAnnotation(table)
                    types = anno.annotation
                    for key, type in types.items():
                        if type == ColumnType.named_entity:
                            Sub_cols_header = [key]
                            break
                if len(Sub_cols_header) != 0:
                    sub_vec = vectors[1][Sub_cols_header, :]
                else:
                    sub_vec = vectors[1]
                vec_table = np.mean(sub_vec, axis=0)
                Z.append(vec_table)

            else:
                vec_table = np.mean(vectors[1], axis=0)
                Z.append(vec_table)
            has_nan = np.isnan(Z).any()
            # if has_nan:
            # print(vectors[0],vectors[1], np.isnan(vec_table).any(),vec_table)

        Z = np.array(Z)
        try:
            clustering_method = ["BIRCH", "Agglomerative"]  # "DBSCAN", "GMM", "KMeans", "OPTICS",
            methods_metrics = {}
            for method in clustering_method:
                print(method)
                metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
                for i in range(0, 6):
                    cluster_dict, metric_dict = clustering_results(Z, T, data_path, ground_truth, method)
                    print("cluster_dict", cluster_dict)
                    metric_df = pd.DataFrame([metric_dict])
                    metric_value_df = pd.concat([metric_value_df, metric_df])
                    dict_file[method + "_" + str(i)] = cluster_dict

                mean_metric = metric_value_df.mean()
                methods_metrics[method] = mean_metric

            e_df = pd.DataFrame()
            for i, v in methods_metrics.items():
                e_df = pd.concat([e_df, v.rename(i)], axis=1)

            store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
            if hp.is_sub is True:
                store_path += "Subject_Col/"
            else:
                store_path += "All/"
            mkdir(store_path)
            e_df.to_csv(store_path + file[:-4] + '_metrics.csv', encoding='utf-8')
            dicts[file] = dict_file
        except ValueError as e:
            continue
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


def column_gts(dataset):
    """gt_clusters: tablename.Columnname:corresponding label dictionary. e.g.{'SOTAB_0.a1': 'Game', 'SOTAB_0.b1': 'date',...}
            ground_t: label and its tables. e.g. {'Game': ['SOTAB_0.a1'], 'Newspaper': ['SOTAB_0.b1', 'SOTAB_0.c1'],...}
            gt_cluster_dict: dictionary of index: label
            like {'Game': 0, 'date': 1, ...}
    """
    groundTruth_file = os.getcwd() + "/datasets/" + dataset + "/column_gtf.xlsx"
    ground_truth_df = pd.read_excel(groundTruth_file, sheet_name=0)
    Superclass = ground_truth_df['Superclass'].unique()
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
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    data_path = os.getcwd() + "/datasets/" + hp.dataset + "/Test/"

    ground_truth = os.getcwd() + "/datasets/" + hp.dataset + "/column_gt.xlsx"
    ground_truth_table = os.getcwd() + "/datasets/" + hp.dataset + "/groundTruth.csv"
    Gt_clusters, Gt_cluster_dict = data_classes(data_path, ground_truth_table)[0], \
                                   data_classes(data_path, ground_truth_table)[2]
    print("Gt_clusters,Gt_cluster_dict", Gt_clusters, Gt_cluster_dict)
    F = open(datafile_path + embedding_file, 'rb')
    content = pickle.load(F)
    # content is the embeddings for all datasets
    Zs = {}
    Ts = {}
    for i in Gt_cluster_dict.keys():
        # gt_cluster_dict {'Event':0} Zs['Event']=[]
        Zs[i] = []
        Ts[i] = []
    column_cluster = column_gts(hp.dataset)[0]

    for vector in content:
        table_name = vector[0]
        table_embeddings = vector[1]
        open_file = data_path + table_name
        try:
            table = pd.read_csv(open_file)
        except (UnicodeDecodeError, TypeError, FileNotFoundError) as e:
            continue
        if len(table_embeddings) != len(table.columns):
            print("Wrong match table embedding and table! Please check!")
            continue
        else:
            for i in range(0, len(table.columns)):
                # gt_clusters ={'SOTAB_0':'Event'}
                classT = Gt_clusters[table_name[0:-4]]
                Ts[classT].append(f"{table_name[0:-4]}.{table.columns[i]}")
                Zs[classT].append(table_embeddings[i])
    clusters_results = {}
    for clu in ['People',
                "Animal"]:  # ["AcademicJournal", "Wrestler", "Plant",  "Newspaper", "Continent", "Animal"]Gt_cluster_dict.keys()
        clusters_result = {}
        Z = Zs[clu]
        T = Ts[clu]
        gt_clusters, ground_t, gt_cluster_dict = columncluster_gt(clu, column_cluster)
        print(gt_clusters, ground_t, gt_cluster_dict)
        if gt_clusters != 0:
            try:
                clustering_method = ["BIRCH", "Agglomerative"]  # "DBSCAN", "GMM", "KMeans", "OPTICS",
                methods_metrics = {}
                store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
                embedding_file_path = embedding_file.split(".")[0]
                col_example_path = os.path.join(store_path, "example", embedding_file_path)
                store_path += "All/" + embedding_file_path + "/"
                mkdir(store_path)
                mkdir(col_example_path)
                for method in clustering_method:
                    print(method)
                    metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
                    for i in range(0, 1):
                        cluster_dict, metric_dict = clusteringColumnResults(Z, T, gt_clusters, ground_t,
                                                                            gt_cluster_dict, method,
                                                                            folderName=col_example_path,
                                                                            filename=f"{clu}.{method}")
                        print("cluster_dict", cluster_dict)
                        if i == 0:
                            clusters_result[method] = cluster_dict
                        metric_df = pd.DataFrame([metric_dict])
                        metric_value_df = pd.concat([metric_value_df, metric_df])

                    mean_metric = metric_value_df.mean()
                    methods_metrics[method] = mean_metric
                    # print("methods_metrics is", methods_metrics)
                clusters_results[clu] = clusters_result
                e_df = pd.DataFrame()
                for i, v in methods_metrics.items():
                    e_df = pd.concat([e_df, v.rename(i)], axis=1)
                e_df.to_csv(store_path + clu + '_metrics.csv', encoding='utf-8')
            except ValueError as e:
                print(e)
    with open(os.path.join(datafile_path, "cluster", embedding_file.split(".")[0] + '_cluster_dict.pickle'),
              'wb') as handle:
        pickle.dump(clusters_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn]
    for file in files:
        starmie_columnClustering(file, hp)
