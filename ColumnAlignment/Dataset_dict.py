import os
import pickle
import random

from datasets import Dataset, DatasetDict
import pandas as pd
import io

# This is for read the pairs in the valentine
def read_valentine_csvs(path_folder):
    dataframes = {}
    for root, dirs, files in os.walk(path_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path, lineterminator='\n')
                dataframes[csv_path] = df
                print(df)
    return dataframes


def dataframe_train(path: str):
    dataframes = {}
    tables = [fn for fn in os.listdir(path) if '.csv' in fn]
    for index, table_name in enumerate(tables):
        fn = os.path.join(path, table_name)
        with open(fn, 'r', newline='', encoding='utf-8') as file:
          data = file.read()
        data = data.replace('\r\n', '\n')
        table = pd.read_csv(io.StringIO(data),lineterminator='\n')
        #table = pd.read_csv(fn,  lineterminator='\r')#lineterminator='\n')
        """if len(table) > 1000:
            table = table.head(1000)"""
        dataframes[table_name] = table
    return dataframes


def col_concate(col: pd.Series, token=False):
    col_string = ''
    if token is True:
        tokens = []
        colVals = [val for entity in col for val in str(entity).split(' ')]
        for val in colVals:
            if val not in tokens:
                tokens.append(val)
                if len(tokens) >= 256:
                    break
        #print(tokens,len(tokens))
        col_string = ' '.join(tokens)
    else:
        col = col.astype(str)
        col_string = col.str.cat(sep=' ')
    return col_string


"slice or random select the cells from the particular columns"


def augmentation_col(column: pd.Series, aug_method: str, aug_percent1=0.5, aug_percent2=0.5, overlap_per=0.5):
    series_1 = pd.Series
    series_2 = pd.Series
    if aug_method == "random":
        total_values = len(column)
        extract_1_percent = int(aug_percent1 * total_values)
        extract_2_percent = int(aug_percent2 * total_values)
        random_values = random.sample(list(column), extract_1_percent)
        series_1 = pd.Series(random_values)

        overlap_count = int(overlap_per * len(series_1))
        overlap_values = random.sample(list(series_1), overlap_count)
        random_values_2 = random.sample(list(column), extract_2_percent)
        series_2 = pd.Series(overlap_values + random_values_2)
        Col1 = col_concate(series_1,True)
        Col2 = col_concate(series_2,True)
    if aug_method == "slice":
        total_values = len(column)
        select_count1 = int(aug_percent1 * total_values)
        series_1 = column.head(select_count1)

        select_count2 = int(aug_percent2 * total_values)
        series_2 = column.tail(select_count2)
    series_1 = series_1.astype(str)
    series_1.str.cat(sep=' ')
    Col1 = col_concate(series_1,True)
    Col2 = col_concate(series_2,True)
    return {'Col1': Col1, 'Col2': Col2, 'label': 1}


def create_column_pairs_mapping(datas: dict):
    dict_all_mapping_positive,dict_all_mapping_negative = [],[]
    dict_pairs_positive, dict_pairs_negative =[],[]
    for tableName, df in datas.items():
        for i in range(len(df.columns)):
            column_i = df.columns[i]
            for percent in [0.1,0.3,0.7,0.9]:
                dict_pairs_positive.append(augmentation_col(df[column_i], "random", 0.6, 0.4,percent))
                dict_all_mapping_positive.append({'Col1': {tableName: column_i}, 'Col2': {tableName: column_i}})
                dict_pairs_positive.append(augmentation_col(df[column_i], "slice", percent, 1-percent))
                dict_all_mapping_positive.append({'Col1': {tableName: column_i}, 'Col2': {tableName: column_i}})

            for j in range(i, len(df.columns)):
                column_j = df.columns[j]
                dict_pairs_negative.append({'Col1': col_concate(df[column_i]), 'Col2': col_concate(df[column_j]), 'label': 0})
                dict_all_mapping_negative.append({'Col1': {tableName: column_i}, 'Col2': {tableName: column_i}})

    dataset_dict = DatasetDict()

    all_pairs_positive = pd.DataFrame(dict_pairs_positive)
    all_pairs_negative = pd.DataFrame(dict_pairs_negative)
    sample_size_positive = int(0.7 * len(all_pairs_positive))
    sample_size_negative = int(0.7 * len(all_pairs_negative))

    train_df_po = all_pairs_positive.sample(n=sample_size_positive, random_state=42)
    eval_df_po = all_pairs_positive.drop(train_df_po.index)

    train_df_ne = all_pairs_negative.sample(n=sample_size_negative, random_state=42)
    eval_df_ne = all_pairs_negative.drop(train_df_ne.index)

    train_dataset = Dataset.from_pandas(pd.concat([train_df_po, train_df_ne])).remove_columns("__index_level_0__")
    eval_dataset = Dataset.from_pandas(pd.concat([eval_df_po, eval_df_ne])).remove_columns("__index_level_0__")
    #print(train_dataset,eval_dataset)
    dataset_dict["train"] = train_dataset
    dataset_dict["eval"] = eval_dataset
    return dataset_dict


def create_column_pairs(pair_path):
    datas = []
    pairs = []
    mapping = []
    tables = [fn for fn in os.listdir(pair_path) if '.csv' in fn]
    if len(tables) != 2:
        print("Wrong schema pair folder! Please check")
        return pairs, mapping

    else:
        for i in range(0, 2):
            fn = os.path.join(pair_path, tables[i])
            with open(fn, 'r', newline='', encoding='utf-8') as file:
                data = file.read()
            data = data.replace('\r\n', '\n')
            table = pd.read_csv(io.StringIO(data),lineterminator='\n')
            datas.append((tables[i], table))
        for column_i in datas[0][1].columns:
            column_pairs = []
            mapping_pairs = []
            for column_j in datas[1][1].columns:
                data1 = datas[0][1]
                data2 = datas[1][1]
                mapping_pairs.append((('table_1', column_i), ('table_2', column_j)))
                column_pairs.append([col_concate(data1[column_i],True), col_concate(data2[column_j],True)])
            pairs.append(column_pairs)
            mapping.append(mapping_pairs)
        return pairs, mapping


def save_dict(dataset_dict, path):
    with open(path, "wb") as f:
        pickle.dump(dataset_dict, f)

"""
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
path = os.path.join(parent_path,"ValentineDatasets/TPC-DI/Semantically-Joinable/Train")
dfs = dataframe_train(path)

dataset_dict = create_column_pairs_mapping(dfs)
print(dataset_dict,len(dataset_dict['train']['label']))
save_dict(dataset_dict,os.path.join(path,"dataset_dict.pkl"))


"""
