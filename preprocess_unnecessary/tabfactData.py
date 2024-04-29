import os
import pickle
import numpy as np
import json
import urllib.error as error
import pandas as pd
import urllib
from Utils import mkdir
from openpyxl.utils.dataframe import dataframe_to_rows

file_path = os.path.join(os.getcwd(), "../datasets", "TabFact", "02TableAttributes.csv")
all_datasets = pd.read_csv(file_path)

# Create a dictionary to store the DataFrames
dataframes = {}
unique_file_names = all_datasets['fileName'].unique()

"""  
for filename in unique_file_names:
    selected_rows = all_datasets[all_datasets['fileName'] == filename]
    new_df = pd.DataFrame(columns=selected_rows['colName'])
    # Iterate over each row in df
    for index, row in selected_rows.iterrows():
        # Get the values from the current row
        col_name = row['colName']
        vals = row['vals']
        vals_list = vals.split(',')

        try:
            new_df[col_name] = vals_list
        except ValueError as e:
            print(filename ,e)
          
url_path = os.path.join(os.getcwd(), "datasets", "TabFact", "table_to_page.json")
# Read JSON data from a file
with open(url_path) as json_file:
    data = json.load(json_file)
target_path  = os.path.join(os.getcwd(), "datasets", "TabFact", "datasets")

for filename, value in data.items():
    url = value[1]
    try:
        tables = pd.read_html(url, flavor='html5lib')
        df = tables[0]
        df.to_csv(os.path.join(target_path,filename))
    except:
        print(filename,url,ValueError,error.HTTPError)
"""

"""for table_name in table_names:
    tableT = tables_sum[tables_sum["fileName"] == table_name]  #
    table = tableT[["colName", "vals"]].set_index("colName").T.reset_index(drop=True)
    table.to_csv(os.path.join(store_path, table_name))
"""

import multiprocessing


def process_item(table_name):
    path = '../datasets/TabFact/02TableAttributes.csv'
    tables_sum = pd.read_csv(os.path.join(os.getcwd(), path))

    store_path = os.path.join(os.getcwd(), "../datasets", "TabFact", "Test")
    tableT = tables_sum[tables_sum["fileName"] == table_name]  #
    table = tableT[["colName", "vals"]].set_index("colName").T.reset_index(drop=True)
    table.to_csv(os.path.join(store_path, table_name),index=False)


if __name__ == "__main__":

    path = '../datasets/TabFact/02TableAttributes.csv'
    tables_sum = pd.read_csv(os.path.join(os.getcwd(), path))
    table_names = tables_sum["fileName"].unique().tolist()
    # Create a multiprocessing Pool with the desired number of processes
    num_processes = multiprocessing.cpu_count()  # Number of CPU cores
    pool = multiprocessing.Pool(num_processes)

    # Execute the for loop in parallel
    pool.map(process_item, table_names)

    # Close the pool to free resources
    pool.close()
    pool.join()
