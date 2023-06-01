import os
import pickle
import random

from datasets import Dataset, DatasetDict
import pandas as pd


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
        table = pd.read_csv(fn, lineterminator='\n')
        if len(table) > 100:
            table = table.head(100)
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
                if len(tokens) >= 512:
                    break
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
    if aug_method == "slice":
        total_values = len(column)
        select_count1 = int(aug_percent1 * total_values)
        series_1 = column.head(select_count1)

        select_count2 = int(aug_percent2 * total_values)
        series_2 = column.tail(select_count2)
    series_1 = series_1.astype(str)
    series_1.str.cat(sep=' ')
    Col1 = col_concate(series_1)
    Col2 = col_concate(series_2)
    return {'Col1': Col1, 'Col2': Col2, 'label': 1}


def create_column_pairs_mapping(datas: dict, aug_meth="random"):
    dict_all_mapping = []
    dict_pairs = []
    for tableName, df in datas.items():
        #print("Table name:", tableName, len(df))
        
        for i in range(len(df.columns)):
            column_i = df.columns[i]
            dict_pairs.append(augmentation_col(df[column_i], aug_meth, 0.5, 0.5, 0.2))
            dict_all_mapping.append({'Col1': {tableName: column_i}, 'Col2': {tableName: column_i}})
            for j in range(i, len(df.columns)):
                column_j = df.columns[j]
                dict_all_mapping.append({'Col1': {tableName: column_i}, 'Col2': {tableName: column_i}})
                dict_pairs.append({'Col1': col_concate(df[column_i]), 'Col2': col_concate(df[column_j]), 'label': 0})
     
    dataset_dict = DatasetDict()
    
    all_pairs = pd.DataFrame(dict_pairs)
    sample_size = int(0.7 * len(all_pairs))
    train_df = all_pairs.sample(n=sample_size, random_state=42)
    eval_df = all_pairs.drop(train_df.index)
    train_dataset =Dataset.from_pandas(train_df).remove_columns("__index_level_0__")
    eval_dataset = Dataset.from_pandas(eval_df).remove_columns("__index_level_0__")
    dataset_dict["train"] = train_dataset
    dataset_dict["eval"] = eval_dataset
    return dataset_dict, dict_all_mapping


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
            table = pd.read_csv(fn, lineterminator='\n')
            datas.append((tables[i], table))
        for column_i in datas[0][1].columns:
            for column_j in datas[1][1].columns:
                data1 = datas[0][1]
                data2 = datas[1][1]
                mapping.append((('table_1', column_i), ('table_2', column_j)))
                pairs.append([col_concate(data1[column_i]), col_concate(data2[column_j])])
        return pairs, mapping


def save_dict(dataset_dict, path):
    with open(path, "wb") as f:
        pickle.dump(dataset_dict, f)


"""path = "D:/CurrentDataset/valentine-data-fabricator/csvfiles/source/"
dfs = dataframe_train(path)
dataset_dict, dict_all_mapping = create_column_pairs_mapping(dfs)
save_dict(dataset_dict,"D:/CurrentDataset/ColumnAlignment/dataset/dataset_dict.pkl")"""
