import os
import pandas as pd
path = os.getcwd()+"/datasets/open_data/Test"
tables = [fn for fn in os.listdir(path) if '.csv' in fn]
for table in tables:
    fn = os.path.join(path, table)
    try:
        table = pd.read_csv(fn,lineterminator='\n')
    except UnicodeDecodeError as e:
        print("Oops!  should delete this file!",table)
        os.remove(os.path.join(path, table))

