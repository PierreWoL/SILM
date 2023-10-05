import numpy as np
import pandas as pd
import random
from d3l.utils.functions import tokenize_str
from TFIDF import table_tfidf, roulette_row_selection,roulette_wheel_selection
import math

def augment(table: pd.DataFrame, op: str):
    """Apply an augmentation operator on a table.

    Args:
        table (DataFrame): the input table
        op (str): operator name
    
    Return:
        DataFrame: the augmented table
    """
    if op == 'drop_col':
        # set values of a random column to 0
        col = random.choice(table.columns)
        table = table.copy()
        table[col] = ""
    elif op == 'sample_row':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5)
    elif op == 'sample_row_TFIDF':
        # sample 50% of rows
        if len(table) > 0:
            table = roulette_row_selection(table, 0.5)
    elif op == 'sample_row_ordered':
        # sample 50% of rows
        if len(table) > 0:
            table = table.sample(frac=0.5).sort_index()
    elif op == 'shuffle_col':
        # shuffle the column orders
        new_columns = list(table.columns)
        random.shuffle(new_columns)
        table = table[new_columns]
    elif op == 'drop_cell':
        # drop a random cell
        table = table.copy()
        row_idx = random.randint(0, len(table) - 1)
        col_idx = random.randint(0, len(table.columns) - 1)
        table.iloc[row_idx, col_idx] = ""
    elif op == 'sample_cells':
        # sample half of the cells randomly
        table = table.copy()
        col_idx = random.randint(0, len(table.columns) - 1)
        sampleRowIdx = []
        for _ in range(len(table) // 2 - 1):
            sampleRowIdx.append(random.randint(0, len(table) - 1))
        for ind in sampleRowIdx:
            table.iloc[ind, col_idx] = ""
    elif op == 'sample_cells_TFIDF':
        table = table.copy()
        table = table.astype(str)
        df_tfidf = table_tfidf(table)
        num_rows = len(table)
        augmented_cols = []
        for col in table.columns:
            selected_indices = roulette_wheel_selection(df_tfidf[col].index, math.ceil((num_rows * 0.5)), df_tfidf[col])
            augmented_col = table[col].iloc[selected_indices].reset_index(drop=True)
            augmented_cols.append(augmented_col)
            # Combine the augmented columns to form the new table
        table = pd.concat(augmented_cols, axis=1)
        del augmented_cols

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
        table = table.astype(str)
        df_tfidf = table_tfidf(table)
        num_rows = len(table)
        for col in table.columns:
            selected_rows = roulette_wheel_selection(df_tfidf[col].index, math.ceil((num_rows * 0.5)), df_tfidf[col])
            # Replace the randomly selected cells with cells picked by roulette_wheel
            for idx in selected_rows:
                table.at[random.choice(table[col].index), col] = table.at[idx, col]
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
        if len(table) < 2:
            table = table.copy()
        else:
            table = table.astype(str)
            tfidf_scores = table_tfidf(table)
            augmented_cols = []
            num_to_retain = math.ceil(len(table)* 0.5)
            for col in table.columns:
                # Sort the column based on TFIDF scores and keep only the top 50%
                sorted_indices = np.argsort(tfidf_scores[col])[::-1][:num_to_retain]
                augmented_col = table[col].iloc[sorted_indices].reset_index(drop=True)
                augmented_cols.append(augmented_col)
            # Combine the augmented columns to form the new table
            table = pd.concat(augmented_cols, axis=1)
            del augmented_cols, tfidf_scores
    elif op == "replace_high_cells":
        table = table.copy()
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
            selected_rows = np.random.choice(sorted_series.index, num_rows // 2, replace=False)
            # Replace the randomly selected cells with high-TFIDF cells
            for idx in selected_rows:
                index_selected = sorted_series.iloc[:len(sorted_series) // 2].index
                table.at[idx, col] = table.at[random.choice(index_selected), col]

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
