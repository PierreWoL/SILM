import os
import pandas as pd

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")


def are_all_numbers(values):
    cleaned_values = [val for val in values if
                      pd.notna(val) and str(val).strip() != '' and val not in ["n / a", "n/a", "N/A"]]
    return all(val.isdigit() for val in cleaned_values)


def aug(table: pd.DataFrame):
    exists = []
    for index in range(0, table.shape[1]):
        if are_all_numbers(table.iloc[:, index][0].split(",")) is False:
            exists.append(index)
    return table.iloc[:,exists]
