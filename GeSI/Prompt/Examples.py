import os
import random

import pandas as pd
import json


# Assuming the file is named 'data.json'

def get_random_elements(data, n):
    # Ensure n is not greater than the length of the list
    if n > len(data):
        return data
    else:
        return random.sample(data, n)


def example(shot=1):
    prompt = 'Prompt/AttributeExample/example.json'
    with open(os.path.join(os.getcwd(), prompt), 'r') as file:
        data = json.load(file)
    data = get_random_elements(data, shot)
    data_copy = []
    for dict_example in data:
        table = pd.read_csv(os.path.join("Prompt/AttributeExample", dict_example['table']))
        header = random.choice(table.columns.tolist())
        select_col = table[header].dropna().tolist()
        k = min(10, len(select_col) - 1)
        if k <= 0:
            col = select_col
        else:
            col = random.choices(select_col, k=k)
        index_col = table.columns.tolist().index(header)
        data_copy.append({'table': table, 'attr': dict_example['attrs'][index_col], 'type': dict_example['type'],
                          'header': header,'path': dict_example['attrs_hier'][index_col],
                          'col': col})
    return data_copy
