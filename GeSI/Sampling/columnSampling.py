import numpy as np
import pandas as pd


def value_frequency(series):
    return series.value_counts().to_dict()


def random_keys_by_frequency(series, num_samples=2):
    freq_dict = value_frequency(series)
    keys = list(freq_dict.keys())
    if num_samples >= len(keys):
        return freq_dict
    # values = list(freq_dict.values())
    # probabilities = [value / len(series) for value in values]
    values = np.asarray(list(freq_dict.values())).astype('float64')
    probabilities = values / np.sum(values)
    selected_keys = np.random.choice(keys, size=num_samples, p=probabilities, replace=False)
    select_dict = {i: freq_dict[i] for i in selected_keys.tolist()}
    return select_dict


"""
# 示例
s = pd.Series(['a', 'b', 'a', 'c', 'a', 'b', 'd', 'd', 'd', 'c'])
print(random_keys_by_frequency(s))"""
