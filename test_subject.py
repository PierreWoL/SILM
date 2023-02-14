import csv
import os
import time

import numpy as np
import pandas as pd
import SubjectColumnDetection as SCD
import TableAnnotation as TA

data_path = os.getcwd() + "/T2DV2/test/"
tables = SCD.datasets(data_path)

example = os.getcwd() + '/T2DV2/test/41648740_0_6959523681065295632.csv'

"""
table = pd.read_csv(example)
print(table)
anno = TA.TableColumnAnnotation(table)
anno.ws_cal(4)
subject_col = anno.subCol(0.1)
"""


# print(subject_col)
# print(table)
# print(key, table_sample)
#
# anno = TA.TableColumnAnnotation(table_sample)
# print(anno.annotation)
#
#


def write_csv(data_row):
    path = os.getcwd() + "/T2DV2/mainColumn.csv"
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


table_subject_columns = {}


def tables_annotate(top_k, table_name, Table, cseid):
    print(table_name)
    anno = TA.TableColumnAnnotation(Table)
    anno.ws_cal(top_k, cseid)
    subject_col = anno.subCol(0.1)
    name_l = [table_name] * len(subject_col)
    output = np.column_stack((np.array(anno.matrix_list), subject_col.T))
    output = np.column_stack((output, np.array(name_l).T))
    column_index = index_of_ones(anno.subject_col)
    table_subject_columns[table_name] = column_index
    """
    for row in output:
        write_csv(list(row))
    """


cse_id = cse_ids[2]
for index, table in tables.items():
        try:
            tables_annotate(4, index, table, cse_id)
        except (TypeError, ValueError):
            pass
        time.sleep(1)
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
