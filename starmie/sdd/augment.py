import numpy as np
import pandas as pd
import random
from d3l.utils.functions import tokenize_str
from TFIDF import compute_avg_tfidf, table_tfidf, roulette_row_selection, roulette_wheel_selection
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

    if op == 'drop_col':
        # set values of a random column to 0
        col = random.choice(table.columns)
        table = table.copy()
        table[col] = ""
    elif op == 'sample_row':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5) if isTabFact is False else sample_TFIDF_rows(table)
    elif op == 'sample_row_TFIDF':
        # sample 50% of rows
        if len(table) > 0:
            table = roulette_row_selection(table, 0.5) if isTabFact is False else sample_TFIDF_rows(table)
    elif op == 'sample_row_ordered':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5).sort_index() if isTabFact is False else sample_TFIDF_rows(table)
    elif op == 'shuffle_col':
        # shuffle the column orders
        new_columns = list(table.columns)
        random.shuffle(new_columns)
        table = table[new_columns]
    elif op == 'drop_cell':
        # drop a random cell
        if isTabFact is False:
            table = table.copy()
            row_idx = random.randint(0, len(table) - 1)
            col_idx = random.randint(0, len(table.columns) - 1)
            table.iloc[row_idx, col_idx] = ""
        else:
            table = table.copy()
            if len(table) > 0:
                col_index = random.choice(range(0, len(table.columns)))
                column_list = split(table.iloc[:, col_index][0])
                random_element_index = column_list.index(random.choice(column_list))
                column_list[random_element_index] = ""
                table.iloc[0, col_index] = pd.Series(column_list).rename(table.columns[col_index])
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
                                                       column_TFIDF.values())
                table.iloc[0, index] = (",").join([list(column_TFIDF.keys())[i] for i in select_ones])
                del column_TFIDF, select_ones



    elif op == 'replace_cells':
        # replace half of the cells randomly with the first values after sorting
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sortedCol = table[table.columns[col_idx]].sort_values().tolist()
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = sortedCol[ind]
    elif op == 'replace_cells_TFIDF':
        # replace half of the cells randomly with values with high tfidf using probabilities
        table = table.copy()
        if isTabFact is False:
            table = table.astype(str)
            df_tfidf = table_tfidf(table)
            num_rows = len(table)
            for col in table.columns:
                selected_rows = roulette_wheel_selection(df_tfidf[col].index, math.ceil((num_rows * 0.5)),
                                                         df_tfidf[col])
                # Replace the randomly selected cells with cells picked by roulette_wheel
                for idx in selected_rows:
                    table.at[random.choice(list(table[col].index)), col] = table.at[idx, col]
        else:
            for index, col in enumerate(table.columns):
                list_col = split(table.iloc[0, index])
                column_TFIDF = compute_avg_tfidf(list_col)
                select_ones = roulette_wheel_selection(range(len(column_TFIDF)), math.ceil((len(column_TFIDF) * 0.5)),
                                                       column_TFIDF.values())
                sorted_column = sorted(column_TFIDF.items(), key=lambda item: item[1], reverse=True)

                for i in select_ones: list_col[i] = sorted_column[0][0]
                table.iloc[0, index] = (",").join(list_col)
                del column_TFIDF, select_ones,sorted_column
    elif op == 'drop_head_cells':
        # drop the first quarter of cells
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sortedCol = table[table.columns[col_idx]].sort_values().tolist()
        sortedHead = sortedCol[:len(table) // 4]
        for ind in range(0, len(table)):
            if table.iloc[ind, col_idx] in sortedHead:
                table.iloc[ind, col_idx] = ""
    elif op == 'drop_num_cells':
        # drop numeric cells
        table = table.copy()
        tableCols = list(table.columns)
        numTable = table.select_dtypes(include=['number'])
        numCols = numTable.columns.tolist()
        if numCols == []:
            col_idx = random.randint(0, len(table.columns) - 1)
        else:
            col = random.choice(numCols)
            col_idx = tableCols.index(col)
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = ""
    elif op == "highlight_cells":
        table = table.copy()
        if isTabFact is False:
            if len(table) > 2:
                table = table.astype(str)
                tfidf_scores = table_tfidf(table)
                augmented_cols = []
                num_to_retain = math.ceil(len(table) * 0.5)
                for col in table.columns:
                    # Sort the column based on TFIDF scores and keep only the top 50%
                    sorted_indices = np.argsort(tfidf_scores[col])[::-1][:num_to_retain]
                    augmented_col = table[col].iloc[sorted_indices].reset_index(drop=True)
                    augmented_cols.append(augmented_col)
                # Combine the augmented columns to form the new table
                table = pd.concat(augmented_cols, axis=1)
                del augmented_cols, tfidf_scores
        else:
            for index, col in enumerate(table.columns):
                list_col = split(table.iloc[0, index])
                column_TFIDF = compute_avg_tfidf(list_col)
                sorted_column = sorted(column_TFIDF.items(), key=lambda item: item[1], reverse=True)
                left  = [i[0] for i in sorted_column if sorted_column.index(i)< math.ceil((len(sorted_column) * 0.5))]
                table.iloc[0, index] = (",").join(left)
                del list_col, column_TFIDF, left, sorted_column
    elif op == "replace_high_cells":
        table = table.copy()
        if isTabFact is False:
            table = table.astype(str)
            tfidf_scores = table_tfidf(table)
            for col in table.columns:
                # Check if the column is in the tfidf_dict
                if col not in tfidf_scores:
                    continue
                # Sort the indices in the column based on their TFIDF values
                # sorted_indices = np.argsort(tfidf_scores[col])[::-1]   # [::-1] is to sort in descending order
                sorted_series = tfidf_scores[col].sort_values(ascending=False)
                # Select half of the rows randomly
                num_rows = len(table[col])
                selected_rows = np.random.choice(sorted_series.index, math.ceil(num_rows * 0.5), replace=False)
                # Replace the randomly selected cells with high-TFIDF cells
                for idx in selected_rows:
                    index_selected = sorted_series.iloc[:math.ceil((len(sorted_series) * 0.5))].index
                    table.at[idx, col] = table.at[random.choice(index_selected), col]
        else:
            for index, col in enumerate(table.columns):
                list_col = split(table.iloc[0, index])

                number = len(list_col)
                selected_rows = np.random.choice(range(number), math.ceil(number * 0.5), replace=False)
                highest_cells = sorted(compute_avg_tfidf(list_col).items(), key=lambda item: item[1], reverse=True)[:math.ceil(number * 0.5)]
                for i in selected_rows: list_col[i] = random.choice(highest_cells)[0]

                table.iloc[0, index] = (",").join(list_col)
                del list_col,  selected_rows, highest_cells

    elif op == 'swap_cells':
        # randomly swap two cells
        table = table.copy()
        row_idx = random.randint(0, len(table) - 1)
        row2_idx = random.randint(0, len(table) - 1)
        while row2_idx == row_idx:
            row2_idx = random.randint(0, len(table) - 1)
        col_idx = random.randint(0, len(table.columns) - 1)
        cell1 = table.iloc[row_idx, col_idx]
        cell2 = table.iloc[row2_idx, col_idx]
        table.iloc[row_idx, col_idx] = cell2
        table.iloc[row2_idx, col_idx] = cell1
    elif op == 'drop_num_col':  # number of columns is not preserved
        # remove numeric columns
        numTable = table.select_dtypes(include=['number'])
        numCols = numTable.columns.tolist()
        textTable = table.select_dtypes(exclude=['number'])
        textCols = textTable.columns.tolist()
        addedCols = 0
        while addedCols <= len(numCols) // 2 and len(numCols) > 0:
            numRandCol = numCols.pop(random.randrange(len(numCols)))
            textCols.append(numRandCol)
            addedCols += 1
        textCols = sorted(textCols, key=list(table.columns).index)
        table = table[textCols]
    elif op == 'drop_nan_col':  # number of columns is not preserved
        # remove a half of the number of columns that contain nan values
        newCols, nanSums = [], {}
        for column in table.columns:
            if table[column].isna().sum() != 0:
                nanSums[column] = table[column].isna().sum()
            else:
                newCols.append(column)
        nanSums = {k: v for k, v in sorted(nanSums.items(), key=lambda item: item[1], reverse=True)}
        nanCols = list(nanSums.keys())
        newCols += random.sample(nanCols, len(nanCols) // 2)
        table = table[newCols]
    elif op == 'shuffle_row':
        # shuffle the rows
        table = table.sample(frac=1)

    return table
