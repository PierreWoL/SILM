import pickle
from typing import Any, Dict, Tuple
import os
import pandas as pd
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def token(text):
    return nltk.word_tokenize(text.lower())

def process_cell(cell):
        if isinstance(cell, str):
            return tuple(token(cell))
        else:
            return (cell,)
class ColToTextTransformer:

    def __init__(self, path, naming_file, tokenizer, max_length: int = 512):
        self.path = path
        self.table_cache = {}
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.column_contexts = column_contexts
        self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        naming_df = pd.read_csv(naming_file)
        self.naming_dict =  naming_df.set_index(naming_df.columns[0])[naming_df.columns[1]].to_dict()
        #naming_dict[table_name]
        for index, table_name in enumerate(self.tables):
            fn = os.path.join(self.path, table_name)
            self.table_cache[table_name]=pd.read_csv(fn)  # encoding="latin-1",

        print(len(self.tables), len(self.table_cache))

        if os.path.isfile(os.path.join(self.path,'tokens.pickle')):
            with open(os.path.join(self.path,'tokens.pickle'), 'rb') as f:
                self.frequency_dictionary = pickle.load(f)
        else:
            self.frequency_dictionary = self._create_frequency_dictionary()


    def _create_frequency_dictionary(self) -> Dict[Any, int]:
        column_cache = []
        unique_cell_values = set()
        for name, table in self.table_cache.items():
            for column_name in table.columns:
                column = table[column_name]
                tokenized_column = [process_cell(cell) for cell in column]
                unique_values_in_column = set(tokenized_column)
                column_cache.append(unique_values_in_column)
                unique_cell_values.update(unique_values_in_column)
            print("table ",name, "unique_cell_values", len(unique_cell_values) )
        frequency_dict = {cell: 0 for cell in unique_cell_values}
        for cell in frequency_dict:
            for token_col in column_cache:
                if cell in token_col:
                    frequency_dict[cell] += 1
        with open(os.path.join(self.path,'tokens.pickle'), 'wb') as f:
            pickle.dump(frequency_dict, f)
        return frequency_dict



    def get_all_column_representations(self, method: str = "title-colname-stat-col",
                                       shuffle_column_values: bool = False) -> Dict[str, Dict[str, str]]:
        column_representations = {}
        n=0
        for table_name, table in self.table_cache.items():
            #print(table_name, table, self.naming_dict[table_name])
            table_column_representations = {}
            for column in table.columns:
                if shuffle_column_values:
                    table[column] = table[column].sample(frac=1).reset_index(drop=True)
                if method == "col":
                    table_column_representations[column] = self.col(table[column])
                elif method == "colname-col":
                    table_column_representations[column] = self.col(table[column], column)
                elif method == "colname-col-context":
                    table_column_representations[column] = self.header_col(table[column], col_name=column)
                elif method == "colname-stat-col":
                    table_column_representations[column] = self.header_stat_col(table[column],
                                                                                              col_name=column)
                elif method == "title-colname-col":
                    table_column_representations[column] = self.title_header_col(table[column],
                                                                                               col_name=column,
                                                                                               table_title=self.naming_dict[table_name])
                elif method == "title-colname-col-context":
                    table_column_representations[column] = self.title_header_col_context(table[column],
                                                                                 col_name=column,
                                                                                 table_title=self.naming_dict[table_name])
                elif method == "title-colname-stat-col":
                    table_column_representations[column] = self.title_header_stat_col(table[column],
                                                                                                    col_name=column,
                                                                                                    table_title=self.naming_dict[table_name])
                else:
                    raise ValueError(f"Method {method} not supported")
                # print("Current column is",column,len(self.tokenizer(table_column_representations[column])),"\n",table_column_representations[column])
            column_representations[table_name] = table_column_representations
            n += 1
            if n == 50:
                break
        return column_representations

    def col(self, column: pd.Series, prefix_text: str = '',
            suffix_text: str = '') -> str:
        column = column.dropna()
        transformed_col, last_value_index = self._concat_until_max_length(column, prefix_text,
                                                                          suffix_text)
        if last_value_index < len(column) - 1:
            column = column.sort_values(key=lambda x: x.map(lambda y: (-self.frequency_dictionary.get(process_cell(y), 0), y)))
            transformed_col, last_value_index = self._concat_until_max_length(column, prefix_text,
                                                                              suffix_text)
        return transformed_col

    def _concat_until_max_length(self, column: pd.Series, initial_transformed_col: str = '',
                                 suffix_to_transformed_col: str = '') -> Tuple[str, int]:
        transformed_col = initial_transformed_col
        last_index = -1
        for i, text in enumerate(column):
            new_text_without_suffix = transformed_col + ', ' + str(text) if i > 0 else transformed_col + ' ' + str(text)
            new_text = new_text_without_suffix + suffix_to_transformed_col
            token_count = len(self.tokenizer(new_text))
            if token_count > self.max_length:
                break
            transformed_col = new_text_without_suffix
            last_index = i
        return transformed_col.strip(), last_index

    def header_col(self, column: pd.Series, col_name: str) -> str:
        transformed_col = col_name + ": "
        return self.col(column, transformed_col)

    def header_col_context(self, column: pd.Series, col_name: str, context: pd.DataFrame) -> str:
        prefix_text = col_name + ": "
        suffix_text = ". " + context
        return self.col(column, prefix_text, suffix_text) + suffix_text

    def header_stat_col(self, column: pd.Series, col_name: str) -> str:
        stats = self._calculate_column_statistics(column)
        prefix_text = f"{col_name} contains {stats[0]} values ({stats[1]}, {stats[2]}, {stats[3]:.2f}): "
        return self.col(column, prefix_text)

    def title_header_col(self, column: pd.Series, col_name: str, table_title: str) -> str:
        prefix_text = f"{table_title}. {col_name}: "
        return self.col(column, prefix_text)

    def title_header_col_context(self, column: pd.Series, col_name: str, table_title: str,
                                 context: str) -> str:
        prefix_text = f"{table_title}. {col_name}: "
        suffix_text = ". " + context
        return self.col(column, prefix_text, suffix_text) + suffix_text

    def title_header_stat_col(self, column: pd.Series, col_name: str, table_title: str) -> str:
        stats = self._calculate_column_statistics(column)
        prefix_text = f"{table_title}. {col_name} contains {stats[0]} values ({stats[1]}, {stats[2]}, {stats[3]:.2f}): "
        return self.col(column, prefix_text)

    def _calculate_column_statistics(self, column: pd.Series) -> Tuple[int, int, int, float]:
        max_length = column.map(lambda x: len(str(x))).max()
        min_length = column.map(lambda x: len(str(x))).min()
        avg_length = column.map(lambda x: len(str(x))).mean()
        return len(column), max_length, min_length, avg_length
