import pandas as pd
import os

f = open(os.getcwd() + "/result/subject_column/subject_column.csv",
         encoding='latin1', errors='ignore')
subcol_CSV = pd.read_csv(f)
for table_name in subcol_CSV.iterrows():
    print()
    tablef = open(os.getcwd() + "/T2DV2/test/" + table_name[1][0] + ".csv",
                  encoding='latin1', errors='ignore')
    table = pd.read_csv(tablef)
    print(table_name[1][1], table)
