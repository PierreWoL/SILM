import os
import pandas as pd
path_f = os.getcwd()+"/datasets/"
datasets = ["SOTAB","T2DV2","Test_corpus"]
folder = ['Test','SubjectColumn']
for dataset in datasets:
    path = os.path.join(path_f, dataset)
    print(path)
    for i in folder:


        tables = [fn for fn in os.listdir(os.path.join(path, i)) if '.csv' in fn]
        for table in tables:
            fn = os.path.join(path, i, table)
            try:
                # print(table)
                table = pd.read_csv(fn, lineterminator='\n')
            except UnicodeDecodeError as e:
                print("Oops!  should delete this file!", table)
                os.remove(os.path.join(path, i, table))
