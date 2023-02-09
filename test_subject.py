import os
import pandas as pd
import SubjectColumnDetection as SCD
import TableAnnotation as TA
data_path = os.getcwd() + "/T2DV2/test/"
tables = SCD.datasets(data_path)

example = '/Users/user/My Drive/CurrentDataset/T2DV2/test/34899692_0_6530393048033763438.csv'
table = pd.read_csv(example)
#table = SCD.random_table(tables)
print(table)
anno = TA.TableColumnAnnotation(table)
#print(anno.annotation)
anno.ws_cal(14)

"""
for i in table.columns:
    detection = SCD.ColumnDetection(table[i])
    typeTest = detection.column_type_judge(2)
    print(detection.col_type)
"""