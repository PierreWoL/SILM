import json
import os

import pandas as pd

'''
         "precision",
         "recall",
         "f1_score"
 "precision_at_10_percent",
        "precision_at_30_percent",
        "precision_at_50_percent",
        "precision_at_70_percent",
        "precision_at_90_percent", 
'''

col = ["Dataset", "M1", "M2","M3","EmbDI", "distribution_based","M1H", "M2H",   "cupid", "similarity_flooding"]
colM = ["Dataset", "RoBERTa_Instance", "SBERT_Instance","RoBERTa_FineTune","EmbDI", "distribution_based","RoBERTa_Schema", "SBERT_Schema",   "cupid", "similarity_flooding"]#,"RoBERTa_FineTune"
col1 = ["Dataset", "M1", "M2","M3","EmbDI", "distribution_based"]#"M3",
col1M = ["Dataset", "RoBERTa_Instance", "SBERT_Instance","RoBERTa_FineTune","EmbDI", "distribution_based"]#"RoBERTa_FineTune",
col2 = ["Dataset","M1H", "M2H",   "cupid", "similarity_flooding"]
col2M = ["Dataset","RoBERTa_Schema", "SBERT_Schema",   "cupid", "similarity_flooding"]
col_meta = ["Dataset_Pair_Name", "Dataset_Pair_relationship",
            "Source_Dataset_shape(Column,Row)", "Target_Dataset_shape(Column,Row)",
            "Number of Ground Truth", 'Number of Column Pairs']


def to_df(column, complex):
    df = pd.DataFrame(columns=column)
    for sublist in complex:
        row = pd.DataFrame([dict(sublist)], columns=column)
        df = pd.concat([df, row])

    return df


data_all = []
dfs_wiki = {}
datasets = ["Magellan", "ChEMBL","TPC-DI", "OpenData"]  # "ChEMBL",, "Magellan" 'Wikidata'


def metadata_true(dataset, data_path,type=''):
    if dataset=='Magellan':
        path = os.path.join(data_path, dataset)
    else:
        path = os.path.join(data_path, dataset,type)

    sub_datasets = [fn for fn in os.listdir(path) if not fn == 'Train' and "." not in fn]
    line_m = []
    for subD in sub_datasets:
        valueaa = [('Dataset_Pair_Name', subD)]
        if dataset == 'Magellan':
            valueaa.append(('Dataset_Pair_relationship', 'Unionable'))
        if dataset == 'Wikidata':
            valueaa.append(('Dataset_Pair_relationship', subD.split("_")[1]))
        else:
            valueaa.append(('Dataset_Pair_relationship', type))
        files = [fn for fn in os.listdir(os.path.join(path, subD)) if not fn == 'results' and
                 not fn == 'config.json']
        column_pairs = {}
        for file in files:
            if file.endswith("source.csv"):
                source_csv = pd.read_csv(os.path.join(path, subD, file))
                column_pairs["Source"] = source_csv.shape[1]
                valueaa.append(('Source_Dataset_shape(Column,Row)', source_csv.shape))
            if file.endswith("target.csv"):
                source_csv = pd.read_csv(os.path.join(path, subD, file))
                column_pairs["Target"] = source_csv.shape[1]
                valueaa.append(('Target_Dataset_shape(Column,Row)', source_csv.shape))
            if file.endswith("_mapping.json"):
                with open(os.path.join(path, subD, file)) as f:
                    data = json.load(f)
                matches_part = data['matches']
                ground_truth = [(match['source_column'], match['target_column']) for match in matches_part]
                valueaa.append(('Number of Ground Truth', len(ground_truth)))
        valueaa.append(('Number of Column Pairs', column_pairs["Target"] * column_pairs["Source"]))
        line_m.append(valueaa)
    dataframe_metadata = to_df(col_meta, line_m)
    return dataframe_metadata


def write_to_xls(dataframes: {pd.DataFrame}, store_path):
    writer = pd.ExcelWriter(store_path)
    for name, df in dataframes.items():
        df.to_excel(writer, sheet_name=name, index=False)
    writer.book.save(store_path)


M_all = ["recall_at_sizeof_ground_truth"]


def recall_k_aggregation(Col_algorithms,Col_algo_M, metr, dataset, data_path):
    """"
    recall summarization for the datasets, not suitable for Wikidata dataset
    """
    line_m = []
    path = os.path.join(data_path, dataset)
    folders = [f for f in os.listdir(path) if
               os.path.isdir(os.path.join(path, f)) and not f.endswith("Train") and not f.endswith(".csv")]
    for folder in folders:
        path_folder = os.path.join(path, folder,"results")

        metrics = [fn[:-12] for fn in os.listdir(path_folder) if '_metrics.csv' in fn]
        valueaa = [('Dataset', folder)]
        for metric in Col_algorithms:
            if metric !='Dataset':
                if metric in metrics:
                    metric_df = pd.read_csv(os.path.join(path_folder, metric + '_metrics.csv'))
                    value = metric_df.loc[metric_df['Key'] == metr, 'Value'].values[0]
                    if metric =='M1':
                        metric = 'RoBERTa_Instance'
                    if metric =='M2':
                        metric = 'SBERT_Instance'
                    if metric =='M3':
                        metric = 'RoBERTa_FineTune'
                    if metric =='M1H':
                        metric = 'RoBERTa_Schema'
                    if metric =='M2H':
                        metric = 'SBERT_Schema'
                    valueaa.append((metric, value))
                else:
                    valueaa.append((metric, ''))
        line_m.append(valueaa)
    data = to_df(Col_algo_M, line_m)
    return data


types = ['Joinable','Semantically-Joinable','Unionable','View-Unionable']  # 'Joinable','Semantically-Joinable','Unionable','View-Unionable'
ab_path = os.path.join(os.getcwd(), '../../ValentineDatasets') #"D:/CurrentDataset/ValentineDatasets/"
for dataset in datasets:
    if dataset == 'Wikidata':
        dataframe_metadata = metadata_true(dataset, ab_path)
    if dataset != 'Wikidata' and dataset!='Magellan':
        parent_path = os.path.join(ab_path, dataset)
        for type in types:
            dfs = {}
            path_sum = os.path.join(ab_path, dataset)
            dataframe_metadata = metadata_true(dataset, ab_path,type)
            df_recall_k = recall_k_aggregation(col,colM,M_all[0],type,path_sum)
            dfs[dataset+"_"+ type+"_basic_info"] = dataframe_metadata
            dfs[dataset + "_" + type+"_1"] = df_recall_k
            dfs[dataset + "_" + type + "_2"] =recall_k_aggregation(col1,col1M,M_all[0],type,path_sum)
            dfs[dataset + "_" + type + "_3"] = recall_k_aggregation(col2,col2M,M_all[0],type,path_sum)
            store_path = os.path.join(ab_path, dataset,type, type + ".xlsx")
            write_to_xls(dfs,store_path)
    else:
        dfs = {}
        path_sum = os.path.join(ab_path, dataset)
        dataframe_metadata = metadata_true(dataset, ab_path)
        df_recall_k = recall_k_aggregation(col,colM,M_all[0], dataset, ab_path)
        dfs[dataset+ "_basic_info"] = dataframe_metadata
        dfs[dataset + "_1"] = df_recall_k
        dfs[dataset + "_2"] = recall_k_aggregation(col1,col1M, M_all[0], '', path_sum)
        dfs[dataset + "_3"] = recall_k_aggregation(col2,col2M, M_all[0], '', path_sum)
        store_path = os.path.join(ab_path, dataset, dataset + ".xlsx")
        write_to_xls(dfs, store_path)






