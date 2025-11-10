"""
This module is augmentation operator.
Based on https://github.com/megagonlabs/starmie/
"""
import numpy as np
import pandas as pd
import random
from learning.TFIDF import table_tfidf, roulette_wheel_selection
import math

def split(column: pd.Series):
    if "," in column:
        return column.split(",")
    elif "|" in column:
        return column.split("|")
    else:
        return column.split(" ")

def augment(table: pd.DataFrame, op: str, percent=0.5, shuffle = False):
    """Apply an augmentation operator on a table.

    Args:
        table (DataFrame): the input table
        op (str): operator name
    
    Return:
        DataFrame: the augmented table
    """

    if op == 'sample_cells':
        # sample percentage of the cells randomly
        table = table.copy()
        # col_idx = random.randint(0, len(table.columns) - 1)
        sampleRowIdx = []
        for _ in range(math.ceil(len(table) * percent)):  # len(table) // 2 - 1
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for col_idx in range(0, len(table.columns)):
            for ind in sampleRowIdx:
                table.iloc[ind, col_idx] = ""

    elif op == 'sample_cells_TFIDF':
        table = table.copy()
        table = table.astype(str)
        df_tfidf = table_tfidf(table)
        num_rows = len(table)
        augmented_cols = []
        for col in table.columns:
            selected_indices = roulette_wheel_selection(df_tfidf[col].index, math.ceil((num_rows * percent)),
                                                        df_tfidf[col])
            augmented_col = table[col].iloc[selected_indices].reset_index(drop=True)
            augmented_cols.append(augmented_col)
            # Combine the augmented columns to form the new table
        table = pd.concat(augmented_cols, axis=1)
        del augmented_cols

    elif op == 'sample_rows':
        table = table.copy()
        if len(table) > 0:
            for index, column in enumerate(table.columns):
                column_list = (",").join(pd.Series(split(table.iloc[:, index][0])).rename(column).sample(frac=0.5))
                table.iloc[0, index] = column_list
    if shuffle is True:
        table = table.sample(frac=1).reset_index(drop=True)
    return table
