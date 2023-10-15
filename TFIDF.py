import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math

from d3l.utils.functions import tokenize_str as tokenize
from typing import Iterable
"""
# Sample list of phrases
phrases = [
    'hello world',
    'goodbye world',
    'hello user',
    'goodbye user',
    'hello',
    'goodbye',
    'user',
    'world'
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True)

# Compute TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(phrases)

# Extract the scores for each phrase
tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=phrases)

#print(tfidf_scores)
avg_tfidf = tfidf_scores.mean(axis=1)
#print(avg_tfidf)
"""


def compute_avg_tfidf(column):
    # Function to compute average TF-IDF for a list of phrases
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(use_idf=True)
    # Compute TF-IDF scores
    if isinstance(column, list):

        try:
            tfidf_matrix = vectorizer.fit_transform(column)
            # Extract the scores for each phrase
            tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(),
                                        index=column)
            mid = tfidf_scores.mean(axis=1)
            result_dict = {}
            for index, value in mid.items():
                if index not in result_dict:
                    result_dict[index] = value
            return result_dict
        except:
            result_dict = {i:1 for i in column}
            return result_dict
    else:
        column_copy = column.copy().apply(tokenize)
        try:
            tfidf_matrix = vectorizer.fit_transform(column_copy)
            tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
            # Compute average TF-IDF scores for each cell
            return tfidf_scores.mean(axis=1).tolist()
        except:
            column_copy[:] = 0
            return column_copy


def table_tfidf(table: pd.DataFrame):
    """
    if len(table)==1:
        result =[]
        for i in table.columns:
            column_list = split(table[i][0])
            i_tfidf = compute_avg_tfidf(column_list)
            result.append(i_tfidf)

        return result
    else:
    """
    # Compute average TF-IDF for each column and store in a new dataframe
    result = table.apply(compute_avg_tfidf)
    return result


def roulette_wheel_selection(index,size, values:Iterable):
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    total = sum(values)
    if total ==0:
        return np.random.choice(index, size=size, replace=False)
    selection_probs = values / total
    non_zero_count = np.count_nonzero(selection_probs)
    if non_zero_count < size:
        return np.random.choice(index, size=size, replace=False)
    return np.random.choice(index, size=size, replace=False, p=selection_probs)


def roulette_row_selection(table, fraction=0.5):
    """
        Select rows using roulette wheel selection method

         df: DataFrame, containing the data you want
         fraction: Ratio of selected rows (between 0-1)
        return: subset of selected rows
    """
    if len(table) == 1:
        return table
    table = table.astype(str)
    df_tfidf = table_tfidf(table)
    df_tfidf['avg_tfidf'] = df_tfidf.mean(axis=1)
    """total_tfidf = sum(df_tfidf['avg_tfidf'])
    probabilities = df_tfidf['avg_tfidf'] / total_tfidf
    num_selections = math.ceil((len(df_tfidf) * fraction))
    selected_indices = np.random.choice(df_tfidf.index, size=num_selections, replace=False, p=probabilities)"""
    selected_indices = roulette_wheel_selection(df_tfidf.index, math.ceil((len(df_tfidf) * fraction)),
                                                df_tfidf['avg_tfidf'])
    del df_tfidf
    return table.loc[selected_indices]
