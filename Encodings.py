from starmie.sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import time
from Utils import mkdir
from argparse import Namespace
import os
import clustering
from Utils import mkdir
from SubjectColumnDetection import ColumnType
import TableAnnotation as TA

def extractVectors(dfs, method, dataset, augment, sample, table_order, run_id, check_subject_Column, singleCol=False):
    ''' Get model inference on tables
    Args:
        dfs (list of DataFrames): tables to get model inference on
        method (str): model saved path folder
        augment (str): augmentation operator used in vector file path (e.g. 'drop_cell')
        sample (str): sampling method used in vector file path (e.g. 'head')
        table_order (str): 'column' or 'row' ordered
        run_id (int): used in file path
        singleCol (boolean): is this for single column baseline
    Return:
        list of features for the dataframe
    '''
    if singleCol:

        model_path = "model/%s/%s/model_%s_%s_%s_%d_%ssingleCol.pt" % (
        method, dataset, augment, sample, table_order, run_id, check_subject_Column)

    else:
        model_path = "model/%s/%s/model_%s_%s_%s_%d_%s.pt" % (
        method, dataset, augment, sample, table_order, run_id, check_subject_Column)
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
        df = pd.read_csv(file, lineterminator='\n', encoding='latin-1')
        if len(df) > 1000:
            # get first 1000 rows
            df = df.head(2000)
            # df = df.sample(n=2000)
        filename = file.split("/")[-1]
        dataDFs[filename] = df
    return dataDFs


def table_features(hp: Namespace):
    DATAFOLDER = hp.dataset
    dataFolder = hp.dataset
    if dataFolder == 'open_data':
        DATAFOLDER = "datasets/open_data/Test/"
    elif dataFolder == 'SOTAB':
        DATAFOLDER = 'datasets/SOTAB/Test/'
    elif dataFolder == 'T2DV2':
        DATAFOLDER = 'datasets/T2DV2/Test/'
    elif dataFolder == 'Test_corpus':
        DATAFOLDER = 'datasets/Test_corpus/Test/'
    tables = get_df(DATAFOLDER)
    print("num dfs:", len(tables))

    dataEmbeds = []
    table_number = len(tables)
    dfs_count = 0
    # Extract model vectors
    cl_features = extractVectors(list(tables.values()), hp.method, hp.dataset, hp.augment_op, hp.sample_meth,
                                 hp.table_order, hp.run_id, hp.check_subject_Column, singleCol=hp.single_column)
    output_path = "result/embedding/%s/vectors/%s" % (hp.method, hp.dataset)
    mkdir(output_path)
    print(output_path)
    for i, file in enumerate(tables):
        dfs_count += 1
        # get features for this file / dataset
        cl_features_file = np.array(cl_features[i])
        dataEmbeds.append((file, cl_features_file))
        # print(len(tables[file].columns),len(cl_features_file), cl_features_file)

    if hp.single_column:
        output_file = "cl_%s_%s_%s_%d_singleCol.pkl" % (hp.augment_op, hp.sample_meth, hp.table_order, hp.run_id)
    else:
        output_file = "cl_%s_%s_%s_%d.pkl" % (hp.augment_op, hp.sample_meth, hp.table_order, hp.run_id)

    output_path += output_file
    if hp.save_model:
        pickle.dump(dataEmbeds, open(output_path, "wb"))


def starmie_clustering(hp: Namespace):
    files = []
    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    data_path = os.getcwd() + "/datasets/"+hp.dataset+"/Test/"
    subject_path = os.getcwd() + "/datasets/"+hp.dataset+"/SubjectColumn/"
    if hp.method == "starmie":
        files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn]
    ground_truth = os.getcwd() + "/datasets/"+hp.dataset+"/groundTruth.csv"
    for file in files:
        F = open(datafile_path + file, 'rb')
        content = pickle.load(F)
        Z = []
        T = []
        for vectors in content:
            T.append(vectors[0][:-4])
            if hp.is_sub is True:
                table = pd.read_csv(data_path + vectors[0],
                                    encoding="latin-1")
                Sub_cols_header = []
                if vectors[0] in [fn for fn in os.listdir(subject_path) if
                                        '.csv' in fn]:
                    Sub_cols = pd.read_csv(subject_path + vectors[0],
                                           encoding="latin-1")
                    for column in Sub_cols.columns.tolist():
                        Sub_cols_header.append(table.columns.tolist().index(column))

                else:
                    anno = TA.TableColumnAnnotation(table)
                    types = anno.annotation
                    # print(types)
                    if ColumnType.named_entity not in types.values():
                        Sub_cols_header = list(types.keys())
                    else:
                        for key, type in types.items():
                            if type == ColumnType.named_entity:
                                Sub_cols_header = [key]
                                break
                sub_vec = vectors[1][Sub_cols_header, :]
                vec_table = np.mean(sub_vec, axis=0)
                Z.append(vec_table)

            else:
                vec_table = np.mean(vectors[1], axis=0)
                Z.append(vec_table)
            has_nan = np.isnan(Z).any()
            #if has_nan:
             #print(vectors[0],vectors[1], np.isnan(vec_table).any(),vec_table)

        Z = np.array(Z)

        clustering_method = ["BIRCH", "Agglomerative"]  # "DBSCAN", "GMM", "KMeans", "OPTICS",
        methods_metrics = {}
        for method in clustering_method:
            print(method)
            metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
            for i in range(0, 3):
                metric_dict = clustering.clustering_results(Z, T, data_path, ground_truth, method)
                metric_df = pd.DataFrame([metric_dict])
                metric_value_df = pd.concat([metric_value_df, metric_df])
            mean_metric = metric_value_df.mean()
            methods_metrics[method] = mean_metric
            #print("methods_metrics is", methods_metrics)
        e_df = pd.DataFrame()
        for i, v in methods_metrics.items():
            #print(v.rename(i))
            e_df = pd.concat([e_df, v.rename(i)], axis=1)
        #print(e_df)
        store_path = os.getcwd() + "/result/" + hp.method + "/" + hp.dataset + "/"
        mkdir(store_path)
        e_df.to_csv(store_path + file[:-4] + '_metrics.csv', encoding='utf-8')


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
