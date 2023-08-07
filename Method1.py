import pickle

import pandas as pd
import os
import experimentalData as ed
import numpy as np
import SubjectColumnDetection as scd
from sentence_transformers import SentenceTransformer
import d3l.utils.functions as fun
from preprocess_unnecessary import JSON
from Utils import mkdir


def SBERT(data_path, feature_csv):
    mkdir(feature_csv)
    T = ed.get_files(data_path)
    exception = []
    table_names = []
    encodings = []
    print(T)
    # index = T.index("Movie_zurichlab.org_September2020_CPA")
    # T = T[index:]
    for table_name in T:
        print(data_path + table_name + ".csv")
        f = open(data_path + table_name + ".csv", errors='ignore')
        table = pd.read_csv(f)
        column = pd.Series([])
        if table.shape[0] > 10000:
            table = table[:6000]
        if table.shape[1] > 1:
            for i in range(0, table.shape[1]):
                if scd.ColumnDetection(table.iloc[:, i]).column_type_judge(3) == scd.ColumnType.named_entity \
                        or scd.ColumnDetection(table.iloc[:, i]).column_type_judge(3) == scd.ColumnType.long_text:
                    column = pd.concat([column, table.iloc[:, i]], axis=0)
        else:
            column = pd.Series(table.iloc[:, 0])
        encoding_col = encoding(column)
        if encoding_col is not None and type(encoding_col) != np.float64:
            JSON.write_line(feature_csv, [table_name, list(encoding_col)])
            table_names.append(table_name)
            encodings.append(list(encoding_col))
        else:
            exception.append(table_name)
    return pd.DataFrame(encodings), table_names, exception


def SBERT_T(data_path, feature_csv):
    T = [fn for fn in os.listdir(data_path)]
    print(T)
    exception = []
    table_names = []
    encodings = []
    print(T)
    #index = T.index("T2DV2_27.csv")
    index = 0
    T = T[index:]
    for table_name in T:
        print(os.path.join(data_path, table_name))
        f = open(os.path.join(data_path, table_name), errors='ignore')
        table = pd.read_csv(f)
        columns = table.columns.tolist()
        """if table.shape[0] > 10000:
           table = table[:200]"""
        """for index, row in table.iterrows():
            value = []
            for column in table.columns:
                value.append(str(column)+" "+str(row[column]))
            rows.append(" ".join(value))"""

            #print(" ".join(value))
        encoding_col = encoding(columns)
        """if encoding_col is not None and type(encoding_col) != np.float64:
            JSON.write_line(feature_csv, [table_name, list(encoding_col)])
            table_names.append(table_name)
            encodings.append(list(encoding_col))
        else:
            exception.append(table_name)"""
        if encoding_col is not None and type(encoding_col) != np.float64:
            encodings.append((table_name,encoding_col))
    with open(feature_csv, 'wb') as handle:
        pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pd.DataFrame(encodings), table_names, exception


def encoding(column):
    model = SentenceTransformer('all-mpnet-base-v2')
    # try:
    column = fun.remove_blank(column)
    column_token = fun.token_list(column)
    column_embeddings = model.encode(column_token)
    average = np.mean(column_embeddings, axis=0)

    return average
    # except ValueError:
    #   return None


# table = pd.read_csv(
#   "/Users/user/My Drive/CurrentDataset/datasets/open_data/SubjectColumn/Higher_Education_UKtable-13.csv")

#samplePath = os.getcwd() + "/datasets/Test_corpus/SubjectColumn/"  # Table/Test/
#SBERT(samplePath, os.getcwd() + "/datasets/Test_corpus/" + "feature.csv")

"""
samplePath = os.getcwd() + "/datasets/SOTAB/SubjectColumn/"  # Table/Test/
SBERT(samplePath, os.getcwd() + "/datasets/SOTAB/" + "feature.csv")

"""

special_tables = ['12193237_0_8699643798888088574', '28086084_0_3127660530989916727',
                  '28646774_0_3256889721737611537', '41480166_0_6681239260286218499', '41839679_0_3009344178474568371',
                  '45533669_1_7463037039059239824', '48805028_0_8933391169600128370', '68779923_1_3240042497463101224']
important = ['28086084_0_3127660530989916727', '41480166_0_6681239260286218499',
             '28646774_0_3256889721737611537', '41839679_0_3009344178474568371'
    , '48805028_0_8933391169600128370']
