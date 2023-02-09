import os
import pandas as pd
import SubjectColumnDetection as SCD
import TableAnnotation as TA
data_path = os.getcwd() + "/T2DV2/test/"
tables = SCD.datasets(data_path)

example = '/Users/user/My Drive/CurrentDataset/T2DV2/test/4501311_8_8306082458935575308.csv'
table = pd.read_csv(example)
#table = SCD.random_table(tables)
#print(table)
anno = TA.TableColumnAnnotation(table)
#print(anno.annotation)
anno.ws_cal(3)
anno.columns_feature()

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