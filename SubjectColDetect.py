import pickle

import TableAnnotation as TA
import os
import pandas as pd

from SubjectColumnDetection import ColumnType

"""
This should be considered running through the embedding steps

"""


def subjectColumns(data_path):
    target = os.path.join(data_path, "SubjectCol.pickle")
    if os.path.exists(target):
        F = open(os.path.join(data_path, "SubjectCol.pickle"), 'rb')
        SE = pickle.load(F)
    else:
        datas = [data for data in os.listdir(os.path.join(data_path, "Test")) if data.endswith("csv")]
        SE = {}
        for data in datas:
            # print(os.path.join(data_path, data))
            table = pd.read_csv(os.path.join(data_path, "Test", data))
            # print(table.transpose())
            anno = TA.TableColumnAnnotation(table)
            types = anno.annotation

            NE_list = [key for key, type in types.items() if type == ColumnType.named_entity]
            type_all = {table.columns[key]: value.name for key, value in types.items()}
            print(data, type_all)
            SE[data] = (NE_list, table.columns, types)

        with open(os.path.join(data_path, 'SubjectCol.pickle'), 'wb') as handle:
            pickle.dump(SE, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return SE
