import pickle

import TableAnnotation as TA
import os
import pandas as pd

from SubjectColumnDetection import ColumnType

"""
This should be considered running through the embedding steps

"""
data_path = os.path.join(os.getcwd(), "datasets/WDC/Test")
datas = [data for data in os.listdir(data_path) if data.endswith("csv")]
SE = {}
for data in datas:
    print(os.path.join(data_path, data))
    table = pd.read_csv(os.path.join(data_path, data))
    # print(table.transpose())
    anno = TA.TableColumnAnnotation(table, isCombine=True)
    types = anno.annotation

    NE_list = [key for key, type in types.items() if type == ColumnType.named_entity]
    """for key, type in types.items():
        if type == ColumnType.named_entity:
            Sub_cols_header = [table.columns[key]]
            break"""

    SE[data] = (NE_list, table.columns, types)
with open(os.path.join(data_path, 'SubjectCol.pickle'), 'wb') as handle:
        pickle.dump(SE, handle, protocol=pickle.HIGHEST_PROTOCOL)
