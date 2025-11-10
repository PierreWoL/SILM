from concurrent.futures import ThreadPoolExecutor
from learning.pretrain import load_checkpoint, inference_on_tables,simplify_string
import torch
import pandas as pd
import numpy as np
import glob
import pickle
from argparse import Namespace
import os
from clustering import data_classes, clusteringColumnResults,  cluster_discovery, cluster_Dict, evaluate_cluster
from Utils import mkdir



def extractVectors(dfs,dataset, augment, lm, sample, table_order, run_id, check_subject_Column,
                   singleCol=False, SubCol=False, header=False, column= False):
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
    print(SubCol)
    op_augment_new =simplify_string(augment)
    model_path = "model/%s/model_%s_lm_%s_%s_%s_%d_%s.pt" % (
            dataset, op_augment_new, lm, sample, table_order, run_id, check_subject_Column)
    if singleCol:
        model_path = "model/%s/model_%slm_%s_%s_%s_%d_%s_singleCol.pt" % (
            dataset, op_augment_new, lm, sample, table_order, run_id, check_subject_Column)
    elif SubCol:
        model_path = "model/%s/model_%s_lm_%s_%s_%s_%d_%s_subCol.pt" % (
            dataset, op_augment_new, lm, sample, table_order, run_id, check_subject_Column)
    elif header:
        model_path = "model/%s/model_%s_lm_%s_%s_%s_%d_%s_header.pt" % (
            dataset, op_augment_new, lm, sample, table_order, run_id, check_subject_Column)
    elif column:
        model_path = "model/%s/model_%s_lm_%s_%s_%s_%d_%s_column.pt" % (
            dataset, op_augment_new, lm, sample, table_order, run_id, check_subject_Column)
            
    print(model_path)
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    # load_checkpoint from sdd/pretain
    model, trainset = load_checkpoint(ckpt)
  
    return inference_on_tables(dfs, model, trainset, batch_size=1024,subject_column=SubCol)


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


def get_df_columns(dataFolder):
    ''' Get the columns of each table in a folder
        Args:
            dataFolder: filepath to the folder with all tables
        Return:
            columnDfs (dict): key is the filename|columnName, value is the column of that table
        '''
    dataFiles = glob.glob(dataFolder + "/*.csv")
    # print(dataFiles)
    columnDfs = {}
    for file in dataFiles:
        df = pd.read_csv(file)  # , lineterminator='\n'
        for col in df.columns:
            # print(df.transpose())
            filename = file.split("/")[-1]
            columnDfs[f"{filename[0:-4]}.{col}"] = pd.DataFrame(df[col])
    return columnDfs



def table_features(hp: Namespace):
    DATAFOLDER = "datasets/%s/Test/" % hp.dataset
    tables = get_df(DATAFOLDER) if hp.column is False else get_df_columns(DATAFOLDER) 
    print("num dfs:", len(tables))

    dataEmbeds = []
    table_number = len(tables)
    dfs_count = 0
    # Extract model vectors
    cl_features = extractVectors(list(tables.values()), hp.dataset, hp.augment_op, hp.lm, hp.sample_meth,
                                 hp.table_order, hp.run_id, hp.check_subject_Column, singleCol=hp.single_column,
                                 SubCol=hp.subject_column, header=hp.header, column=hp.column)
    output_path = "result/embedding/%s/" % (hp.dataset)
    op_augment_new = simplify_string(str(hp.augment_op))
    mkdir(output_path)
    for i, file in enumerate(tables):
        dfs_count += 1
        # get features for this file / dataset
        cl_features_file = np.array(cl_features[i])
        dataEmbeds.append((file, cl_features_file))
        if i<3:
          print(len(tables[file].columns), len(cl_features_file))

        # print(len(tables[file].columns),len(cl_features_file), cl_features_file)

    output_file = "cl_%s_lm_%s_%s_%s_%d_%s.pkl" % (op_augment_new, hp.lm, hp.sample_meth,
                                                   hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.single_column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_singleCol.pkl" % (op_augment_new, hp.lm, hp.sample_meth,
                                                                 hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.subject_column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_subCol.pkl" % (op_augment_new, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.header:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_header.pkl" % (op_augment_new, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)
    if hp.column:
        output_file = "cl_%s_lm_%s_%s_%s_%d_%s_column.pkl" % (op_augment_new, hp.lm, hp.sample_meth,
                                                              hp.table_order, hp.run_id, hp.check_subject_Column)

    output_path += output_file
    if hp.save_model:
        pickle.dump(dataEmbeds, open(output_path, "wb"))



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

