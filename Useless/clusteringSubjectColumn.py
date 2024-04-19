import pandas as pd
import os
import ast
import d3l.indexing.feature_extraction.values.fasttext_embedding_transformer as fast
import d3l.utils.functions as fun
import SubjectColumnDetection as scd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
import numpy as np
subjectColumn = os.getcwd()+ "/result/subject_column/subject_column.csv"
f = open(subjectColumn, encoding='latin1', errors='ignore')
gt_CSV = pd.read_csv(f)
dict_columns = {}
table_features = {}
table_feature = "result/subject_column/test_subject_column.csv"
ft = open(subjectColumn, encoding='latin1', errors='ignore')
features_csv = pd.read_csv(ft)
"""
for row in gt_CSV.iterrows():
    dict_columns[row[1].values[0]] = row[1].values[1]
    f_table = open(os.getcwd()+"/T2DV2/test/"+row[1].values[0]+".csv", errors='ignore')
    table = pd.read_csv(f_table)
    table_subject = table.iloc[:, ast.literal_eval(row[1].values[1])]
    if scd.ColumnDetection(table_subject.iloc[:,0]).column_type_judge(3) !=scd.ColumnType.named_entity:
            continue
    table_subject.to_csv('result/subject_column/test/'+row[1].values[0]+".csv", encoding='utf-8', index=False)

"""
"""
for row in gt_CSV.iterrows():
    dict_columns[row[1].values[0]] = row[1].values[1]
    f_table = open(os.getcwd()+"/T2DV2/test/"+row[1].values[0]+".csv", errors='ignore')
    table = pd.read_csv(f_table)
    table_subject = table.iloc[:, ast.literal_eval(row[1].values[1])]
    if scd.ColumnDetection(table_subject.iloc[:,0]).column_type_judge(3) !=scd.ColumnType.named_entity:
            continue
    tokens = table_subject.applymap(lambda x:fun.token_stop_word(str(x)))

    try:
        transformer_fast = fast.FasttextTransformer()
        fast_feature = transformer_fast.transform_fast(tokens)
        table_features[row[1].values[0]] = fast_feature
    except ValueError:
        continue

"""
