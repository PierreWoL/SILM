import os
import pickle
from argparse import Namespace
import time
import pandas as pd

from clustering import data_classes


def matrix(table_list: list, table_path):
    start_time = time.time()
    table_pairs = {}
    # initialize the matrix for random two tables m x n
    for table_i in table_list:
        table_i_headers = pd.read_csv(os.path.join(table_path, table_i + ".csv"))
        # print(table_i, table_i_headers.columns.tolist())
        start_j = table_list.index(table_i)
        for table_j in table_list[start_j + 1:]:
            table_j_headers = pd.read_csv(os.path.join(table_path, table_j + ".csv"))
            # print(table_j, table_j_headers.columns.tolist())
            Distinct_cols_i = [table_i + "." + i for i in table_i_headers.columns.tolist()]
            Distinct_cols_j = [table_j + "." + i for i in table_j_headers.columns.tolist()]
            table_pairs[(table_i, table_j)] = {'MatchCol': [], 'Distinct': Distinct_cols_i + Distinct_cols_j,
                                               'Matchscore': 0}
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds for initializing matrix")
    return table_pairs


def JaccardMatrix(col_clusters: dict, table_path):
    table_pairs = {}
    # initialize the matrix for random two tables m x n
    start_time = time.time()
    clusters_tables = {}
    for index, cluster in col_clusters.items():
        clusters_tables[index] = {}
        for columns in cluster:
            table_name = columns.split("html.")[0] + "html" if "html" in columns else columns.split(".")[0]
            column_name = columns.split("html.")[1] if "html" in columns else columns.split(".")[1]
            if table_name not in clusters_tables[index].keys():
                clusters_tables[index][table_name] = [column_name]
            else:
                clusters_tables[index][table_name].append(column_name)
        table_in_cluster = list(clusters_tables[index].keys())
        for table_i in table_in_cluster:
            start_j = table_in_cluster.index(table_i)
            cols_i = clusters_tables[index][table_i]
            for table_j in table_in_cluster[start_j + 1:]:
                cols_j = clusters_tables[index][table_j]
                if (table_i, table_j) in table_pairs.keys():
                    updateOperation(table_pairs, table_i, table_j, cols_i, cols_j)
                    continue
                elif (table_j, table_i) in table_pairs.keys():
                    updateOperation(table_pairs, table_j, table_i, cols_i, cols_j)
                    continue
                elif (table_i, table_j) not in table_pairs.keys() and (table_j, table_i) not in table_pairs.keys():
                    InsertOperation(table_pairs, table_i, table_j, cols_i, cols_j, table_path)
                    continue
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds for matrix")
    # print(f"searching Complete! { table_pairs}")
    jaccard_score = {}
    for pairs in table_pairs.keys():
        jaccard_score[pairs] = 1 - len(table_pairs[pairs]['MatchCol']) / \
                               (len(table_pairs[pairs]['Distinct']) + table_pairs[pairs]['Matchscore'])
    return clusters_tables, table_pairs,jaccard_score


def updateOperation(table_pairs, table_i, table_j, cols_i, cols_j):
    # print(f"Update! {(table_i, table_j)}. {table_pairs[(table_i, table_j)]['Matchscore']} ")
    all_matched = cols_i + cols_j
    for col in all_matched:
        table_pairs[(table_i, table_j)]['MatchCol'].append(col)
    table_pairs[(table_i, table_j)]['Matchscore'] += 1

    table_pairs[(table_i, table_j)]['Distinct'] = [ele for ele in table_pairs[(table_i, table_j)]['Distinct']
                                                   if ele not in table_pairs[(table_i, table_j)]['MatchCol']]
    # print(f"Update finish! {(table_i, table_j)}. {table_pairs[(table_i, table_j)]['Matchscore']} ")


def InsertOperation(table_pairs, table_i, table_j, cols_i, cols_j, table_path):
    all_matched = cols_i + cols_j
    table_i_headers = pd.read_csv(os.path.join(table_path, table_i + ".csv"))
    table_j_headers = pd.read_csv(os.path.join(table_path, table_j + ".csv"))
    Distinct_cols_i = [table_i + "." + i for i in table_i_headers.columns.tolist()]
    Distinct_cols_j = [table_j + "." + i for i in table_j_headers.columns.tolist()]
    table_pairs[(table_i, table_j)] = {'MatchCol': all_matched, 'Matchscore': 1,
                                       'Distinct': Distinct_cols_i + Distinct_cols_j
                                       }
"""

embedding_file_path = "cl_drop_num_col_lm_roberta_head_column_0_subjectheader"
datafile_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "result/starmie/TabFact",
                             "All/" + embedding_file_path + "/column")
# datafile_path = os.path.join(os.getcwd(),"result/starmie/TabFact","All/" +  embedding_file_path+ "/column")
ground_truth_table = os.path.abspath(os.path.dirname(os.getcwd())) + "/datasets/TabFact/groundTruth.csv"
data_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/datasets/TabFact/Test/"
Gt_clusters, Ground_t, Gt_cluster_dict = data_classes(data_path, ground_truth_table)

target_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/result/Valerie/Column/TabFact/_gt_cluster.pickle"
F_cluster = open(target_path, 'rb')
KEYS = pickle.load(F_cluster)

files = [fn for fn in os.listdir(datafile_path) if fn.endswith('.pickle')]
inx = 12
example_file = files[inx]  # 12
F_cluster = open(os.path.join(datafile_path, example_file), 'rb')
col_cluster = pickle.load(F_cluster)
index_cols = example_file = int(files[inx].split("_")[0])
tables = Ground_t[KEYS[index_cols]]
print(tables)
# table_pair = matrix(tables, data_path)
jaccard_score = JaccardMatrix(col_cluster["Agglomerative"], data_path)[2]
print(jaccard_score)
"""