"""
This module is augmentation operator.
Based on https://github.com/megagonlabs/starmie/
"""
import numpy as np
import pandas as pd
import random
from TFIDF import compute_avg_tfidf, table_tfidf, roulette_wheel_selection
import math
from Utils import split


def augment(table: pd.DataFrame, op: str, isTabFact=False):
    """Apply an augmentation operator on a table.

    Args:
        table (DataFrame): the input table
        op (str): operator name
    
    Return:
        DataFrame: the augmented table
    """

    def sample_TFIDF_rows(table: pd.DataFrame):
        table = table.copy()
        if len(table) > 0:
            for index, column in enumerate(table.columns):
                column_list = (",").join(pd.Series(split(table.iloc[:, index][0])).rename(column).sample(frac=0.5))
                table.iloc[0, index] = column_list
        return table


    if op == 'sample_cells':
        # sample half of the cells randomly
        if isTabFact is False:
            table = table.copy()
            col_idx = random.randint(0, len(table.columns) - 1)
            sampleRowIdx = []
            for _ in range(len(table) // 2 - 1):
                sampleRowIdx.append(random.randint(0, len(table) - 1))
            for ind in sampleRowIdx:
                table.iloc[ind, col_idx] = ""
        else:
            table = table.copy()
            col_index = random.choice(range(0, len(table.columns)))
            column_list = split(str(table.iloc[:, col_index][0]))
            num_to_change = len(table) // 2
            if num_to_change < 1:
                num_to_change = 1
            indices_to_change = random.sample(range(len(column_list)), num_to_change)
            # print(indices_to_change)
            for index in indices_to_change:
                # column_list[index] = ""
                del column_list[index]
            table.iloc[0, col_index] = pd.Series(column_list).rename(table.columns[col_index])
    elif op == 'sample_cells_TFIDF':
        table = table.copy()
        if isTabFact is False:
            table = table.astype(str)
            df_tfidf = table_tfidf(table)
            num_rows = len(table)
            augmented_cols = []
            for col in table.columns:
                selected_indices = roulette_wheel_selection(df_tfidf[col].index, math.ceil((num_rows * 0.5)),
                                                            df_tfidf[col])
                augmented_col = table[col].iloc[selected_indices].reset_index(drop=True)
                augmented_cols.append(augmented_col)
                # Combine the augmented columns to form the new table
            table = pd.concat(augmented_cols, axis=1)
            del augmented_cols
        else:
            for index, col in enumerate(table.columns):
                list_col = split(table.iloc[0, index])
                column_TFIDF = compute_avg_tfidf(list_col)
                select_ones = roulette_wheel_selection(range(len(column_TFIDF)), math.ceil((len(column_TFIDF) * 0.5)),
                                                       column_TFIDF.values())
                table.iloc[0, index] = (",").join([list(column_TFIDF.keys())[i] for i in select_ones])
                del column_TFIDF, select_ones

    """
     elif op == 'sample_cells':
        # sample half of the cells randomly
        if isTabFact is False:
            table = table.copy()
            col_idx = random.randint(0, len(table.columns) - 1)
            sampleRowIdx = []
            for _ in range(len(table) // 2 - 1):
                sampleRowIdx.append(random.randint(0, len(table) - 1))
            for ind in sampleRowIdx:
                table.iloc[ind, col_idx] = ""
        else:
            table = table.copy()
            col_index = random.choice(range(0, len(table.columns)))
            column_list = split(table.iloc[:, col_index][0])
            num_to_change = len(column_list) // 2
            indices_to_change = random.sample(range(len(column_list)), num_to_change)
            for index in indices_to_change:
                column_list[index] = ""
            table.iloc[0, col_index] = pd.Series(column_list).rename(table.columns[col_index])
    elif op == 'sample_cells_TFIDF':
        table = table.copy()
        if isTabFact is False:
            table = table.astype(str)
            df_tfidf = table_tfidf(table)
            num_rows = len(table)
            augmented_cols = []
            for col in table.columns:
                selected_indices = roulette_wheel_selection(df_tfidf[col].index, math.ceil((num_rows * 0.5)),
                                                            df_tfidf[col])
                augmented_col = table[col].iloc[selected_indices].reset_index(drop=True)
                augmented_cols.append(augmented_col)
                # Combine the augmented columns to form the new table
            table = pd.concat(augmented_cols, axis=1)
            del augmented_cols
        else:
            for index, col in enumerate(table.columns):
                list_col = split(table.iloc[0, index])
                column_TFIDF = compute_avg_tfidf(list_col)

                select_ones = roulette_wheel_selection(range(len(column_TFIDF)), math.ceil((len(column_TFIDF) * 0.5)),
                                                       column_TFIDF)
                table.iloc[0, index] = (",").join([list_col[i] for i in select_ones])
                del column_TFIDF, select_ones

    
    """

    return table
