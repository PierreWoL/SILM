import os
import time
import pandas as pd



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
    """

        Parameters
        ----------
        col_clusters : dict{cluster id: [T1.A1, T2.A2 , ..., TnAn]}
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        table_path : str
            The path that stores all the datasets

        Return
        ----------
        clusters_tables: the dict contains the Jaccard matrix needed information
        table_pairs:
        jaccard_score:
    """
    table_names = os.listdir(table_path)
    table_pairs = {}
    # initialize the matrix for random two tables m x n
    start_time = time.time()
    clusters_tables = {}
    for index, cluster in col_clusters.items():
        clusters_tables[index] = {}
        for columns in cluster:
            if columns.count('.')>1:
                table_name = columns.split(".")[0]
                for item in table_names:
                    if table_name in item:
                        table_name = item.split(".csv")[0]
                column_name =columns.replace(table_name+".", '', 1)
                #print(f" {columns} table { table_name} column {column_name}")
            else:
                table_name = columns.split(".")[0]
                column_name = columns.split(".")[1]
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
    #print(f"Elapsed time: {elapsed_time:.4f} seconds for matrix")
    # print(f"searching Complete! { table_pairs}")
    jaccard_score = {}
    for pairs in table_pairs.keys():
        jaccard_score[pairs] = 1 - table_pairs[pairs]['Matchscore']/ \
                               (len(table_pairs[pairs]['Distinct']) + table_pairs[pairs]['Matchscore'])
        # len(table_pairs[pairs]['MatchCol']) /  (len(table_pairs[pairs]['Distinct']) + table_pairs[pairs]['Matchscore'])
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
                                       'Distinct': Distinct_cols_i + Distinct_cols_j}
