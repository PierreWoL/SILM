import nltk
from nltk.util import ngrams
from collections import Counter
import math
import pandas as pd

# Column of cells
column = ["This is a sentence.", "This is another sentence.", "A third sentence."]

# Pre-processing
column = [cell.lower() for cell in column]
column = [nltk.word_tokenize(cell) for cell in column]

# Calculate the bigrams
bigrams = [list(ngrams(cell, 2)) for cell in column]

# Calculate the bag-of-words representation
bow_representation = [Counter(cell) for cell in bigrams]

print(bow_representation)


def add_colon_zero(cell):
    return "{}: 0".format(cell)


df = pd.DataFrame({'column_1': ['a', 'b', 'c']})
df['column_1'] = df.applymap(add_colon_zero)
print(df)

import pandas as pd

# Sample dataframe
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})

# Index to start deleting from
i = 2

# Delete rows from index i to the end
df = df.drop(index=df.index[i:])
print(df)
"""

import math

def calculate_entropy(key_value_pairs):
    entropy = 0
    total = sum(key_value_pairs.values())
    for value in key_value_pairs.values():
        entropy -= value/total * math.log(value/total)
    return entropy

def have_converged(key_value_pairs, threshold=0.0001):
    previous_entropy = calculate_entropy(key_value_pairs)
    while True:
        # Perform an iteration of the algorithm
        # ...

        current_entropy = calculate_entropy(key_value_pairs)
        if abs(current_entropy - previous_entropy) < threshold:
            return True
        previous_entropy = current_entropy
        
        
        
        
def has_converged(prev_pairs,cur_pairs, tolerance=1e-6):
    while True:
        for key in d:
            d[key] = update_value(d, key)
        current_values = list(d.values())
        entropy_prev = entropy(prev_values)
        entropy_current = entropy(current_values)
        if abs(entropy_prev - entropy_current) < tolerance:
            return True
        prev_values = current_values

def entropy(values):
    total = sum(values)
    entropy = 0
    for value in values:
        p = value / total
        entropy += -p * math.log2(p) if p != 0 else 0
    return entropy


def calculate_entropy(key_value_pair:tuple):
    entropy = 0
    total = key_value_pair[1]
    for value in key_value_pairs.values():
        p = value / total
        entropy -= p * math.log(p)
    return entropy





"""
