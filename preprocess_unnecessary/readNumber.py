import io
import os
import sys
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file = os.path.join(os.getcwd(), "../datasets/WDC/column_gtf_number.xlsx")
csv = pd.read_excel(file)
file_path = os.path.join(os.getcwd(), "../datasets/WDC/Test")
table_names = os.listdir(file_path)

"""tables = csv["Source_Dataset"].unique()
for table_name in tables:
    table = pd.read_csv(os.path.join(file_path, table_name+".csv"),encoding="UTF-8")
    filtered_rows = csv[csv.iloc[:, 0] == table_name]
    for index, row in filtered_rows.iterrows():
        print("table", table_name,index,str(row["column1"]))
        print(table[str(row["column1"])][:10])


csv.groupby("Table_Cluster_Label")
table_labels = csv["Table_Cluster_Label"].unique()
for labels in table_labels:
    
for index, row in csv.iterrows():
    print(index, row)
"""
