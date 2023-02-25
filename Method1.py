import pandas as pd
import os
import experimentalData as ed
import numpy as np
import SubjectColumnDetection as scd
from sentence_transformers import SentenceTransformer
import d3l.utils.functions as fun


def SBERT(data_path):
    T = ed.get_files(data_path)
    exception = []
    table_names = []
    encodings = []
    for table_name in T:
        f = open(data_path + table_name + ".csv", errors='ignore')
        table = pd.read_csv(f, header=None)
        if table.shape[1] > 1:
            if table_name == "21329809_0_5526008408364682899" or table_name == "26310680_0_5150772059999313798":
                table = table.iloc[:, 1]
            else:
                table = table.iloc[:, 0]
        type_column = scd.ColumnDetection(table[0]).column_type_judge(3)
        encoding_col = encoding(table)
        if encoding_col is not None:
            table_names.append(table_name)
            encodings.append(list(encoding_col))
        else:
            exception.append(table_name)
    return pd.DataFrame(encodings), table_names, exception


def remove_blank(column: pd.Series):
    for index, item in column.items():
        if fun.is_empty(item):
            column.drop([index])
    return list(column)


def encoding(column):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    try:
        column = remove_blank(column)
        column_embeddings = model.encode(column)
        average = np.mean(column_embeddings, axis=0)
        return average
    except ValueError:
        return None


special_tables = ['12193237_0_8699643798888088574', '28086084_0_3127660530989916727',
                  '28646774_0_3256889721737611537', '41480166_0_6681239260286218499', '41839679_0_3009344178474568371',
                  '45533669_1_7463037039059239824', '48805028_0_8933391169600128370', '68779923_1_3240042497463101224']
important = ['28086084_0_3127660530989916727','41480166_0_6681239260286218499',
             '28646774_0_3256889721737611537','41839679_0_3009344178474568371'
             ,'48805028_0_8933391169600128370']
"""
for table_name in special_tables:
    f = open(samplePath + table_name + ".csv", errors='ignore')
    table = pd.read_csv(f, header=None)
    type_column = scd.ColumnDetection(table[0]).column_type_judge(3)
"""
