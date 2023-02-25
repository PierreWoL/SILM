import csv
import os
import time
import numpy as np
import SubjectColumnDetection as SCD
import TableAnnotation as TA

"""
table = pd.read_csv(example)
print(table)
anno = TA.TableColumnAnnotation(table)
anno.ws_cal(4)
subject_col = anno.subCol(0.1)
"""

"""
This dataset is for annotating all table's column
"""


# print(subject_col)
# print(table)
# print(key, table_sample)
#
# anno = TA.TableColumnAnnotation(table_sample)
# print(anno.annotation)
#
#


def write_csv(data_row, filepath):
    path = os.getcwd() + filepath
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


def slice_dict(dict_tables: dict, start, end):
    keys = dict_tables.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = dict_tables[k]
    return dict_slice


cse_ids = ["87f5671ca9e2242a9",
           "14d31f423f0e9466a",
           "f1ebf42224d5144e1",
           "d06ed446ffb7b4a08",
           "a405e50e83c3a4d51",
           "c08f6c0abe4964877",
           "94034e15906404cb1"]


def index_of_ones(lst):
    return [ind for ind, x in enumerate(lst) if x == 1]


def table_annotate(top_k, table_name, Table, cseid, feature_csv):
    print(table_name,Table)
    table_subject_columns = {}
    anno = TA.TableColumnAnnotation(Table)
    anno.ws_cal(top_k, cseid)
    subject_col = anno.subCol(0.05)
    name_l = [table_name] * len(subject_col)
    output = np.column_stack((np.array(anno.matrix_list), subject_col.T))
    output = np.column_stack((output, np.array(name_l).T))
    column_index = index_of_ones(anno.subject_col)
    table_subject_columns[table_name] = column_index
    for row in output:
        write_csv(list(row), feature_csv)


def tables_annotate(datapath, feature_csv):
    tables = SCD.datasets(datapath)
    cse_id = "c08f6c0abe4964877"
    keys = list(tables.keys())
    index_s = keys.index("Library_khocacom.vn_September2020")
    print([index_s])
    table1 = dict([(key, tables[key]) for key in keys[index_s+1:]])

    for index, table in table1.items():
        try:
            table_annotate(4, index, table, cse_id, feature_csv)
        except (TypeError, ValueError):
            pass
        time.sleep(0.3)


data_path = os.getcwd() + "/datasets/WDC_corpus/"
#data_path = os.getcwd() + "/datasets/SOTAB/Table"
#data_path = os.getcwd() +"/datasets/WDC_corpus/"
#tables = SCD.datasets(data_path)
tables_annotate(data_path, "/datasets/features/WDC_corpus_feature.csv")
"""
data_path = os.getcwd() + "/T2DV2/test/"
tables = SCD.datasets(data_path)
example = os.getcwd() + '/T2DV2/test/41648740_0_6959523681065295632.csv'
cse_id = cse_ids[2]
for index, table in tables.items():
        try:
            tables_annotate(4, index, table, cse_id)
        except (TypeError, ValueError):
            pass
        time.sleep(1)

"""

"""

for i in list(range(8)):
    start = 30 * i
    end = 30 * i + 29
    if end >= len(tables):
        end = len(tables) - 1
    slice_tables = slice_dict(tables, start, end)
    cse_id = cse_ids[i]
    for index, table in slice_tables.items():
        try:
            tables_annotate(4, index, table, cse_id)
        except (TypeError, ValueError):
            pass
        time.sleep(1)

"""
# tables_annotate(index, table, cse_id)

"""
NOTE the web search is very unstable 
and sometimes it can turn the unrelated named-entity table 
with very high scores and I don't know why
"""
"""
for i in table.columns:
    detection = SCD.ColumnDetection(table[i])
    typeTest = detection.column_type_judge(2)
    print(detection.col_type)
"""
