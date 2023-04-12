from starmie.sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import time
from Utils import mkdir
from argparse import Namespace


def extractVectors(dfs, method, augment, sample, table_order, run_id, singleCol=False):
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

        model_path = "model/%s/model_%s_%s_%s_%dsingleCol.pt" % (method, augment, sample, table_order, run_id)

    else:
        model_path = "model/%s/model_%s_%s_%s_%d.pt" % (method, augment, sample, table_order, run_id)
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
    cl_features = extractVectors(list(tables.values()), hp.method, hp.augment_op, hp.sample_meth,
                                 hp.table_order, hp.run_id, singleCol=hp.single_column)
    output_path = "result/embedding/%s/vectors/" % (hp.method)
    mkdir(output_path)
    for i, file in enumerate(tables):
        dfs_count += 1
        # get features for this file / dataset
        cl_features_file = np.array(cl_features[i])
        dataEmbeds.append((file, cl_features_file))
        print((file, len(cl_features_file)))

    if hp.single_column:
        output_file  ="cl_%s_%s_%s_%d_singleCol.pkl" %(hp.augment_op, hp.sample_meth, hp.table_order, hp.run_id)
    else:
        output_file = "cl_%s_%s_%s_%d.pkl" %  ( hp.augment_op, hp.sample_meth, hp.table_order, hp.run_id)

    output_path += output_file
    if hp.save_model:
        pickle.dump(dataEmbeds, open(output_path, "wb"))
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
