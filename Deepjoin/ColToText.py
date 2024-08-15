import random
from typing import Dict, Tuple
import os
import pandas as pd
import nltk

from Utils import childList
from d3l.utils.functions import tokenize_with_number


def token(text):
    return nltk.word_tokenize(text.lower())


def process_cell(cell):
    if isinstance(cell, str):
        return tuple(token(cell))
    else:
        return (cell,)


def _calculate_column_statistics(column: pd.Series) -> Tuple[int, int, int, float]:
    max_length = column.map(lambda x: len(str(x))).max()
    min_length = column.map(lambda x: len(str(x))).min()
    avg_length = column.map(lambda x: len(str(x))).mean()
    return len(column), max_length, min_length, avg_length


class ColToTextTransformer:

    def __init__(self, path,  tokenizer, naming_file="",max_length: int = 512, shuffle=False,select=200):
        self.path = path
        self.table_cache = {}
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.column_contexts = column_contexts
        self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        self.tables = childList(self.tables,select )
        if naming_file == "" or os.path.exists(naming_file) is False:
            naming_df = pd.read_csv(naming_file)
            self.naming_dict = naming_df.set_index(naming_df.columns[0])[naming_df.columns[1]].to_dict()
        else:
            self.naming_dict = None
        self.shuffle = shuffle
        # naming_dict[table_name]
        for index, table_name in enumerate(self.tables):
            fn = os.path.join(self.path, table_name)
            self.table_cache[table_name] = pd.read_csv(fn)  # encoding="latin-1",

        print(len(self.tables), len(self.table_cache))

    def get_all_column_representations(self, method: str = "title-colname-stat-col"):
        column_representations = {}
        shuffled_representations = {}
        n = 0
        for table_name, table in self.table_cache.items():
            print(n, table_name, self.naming_dict[table_name])
            table_column_representations = {}
            shuffled_column_representations = {}
            for column in table.columns:
                text, shuffleText = "", ""
                if method == "col":
                    text, shuffleText = self.col(table[column])
                elif method == "colname-col":
                    text, shuffleText = self.col(table[column], column)
                elif method == "colname-col-context":
                    text, shuffleText = self.header_col(table[column], col_name=column)
                elif method == "colname-stat-col":
                    text, shuffleText = self.header_stat_col(table[column],
                                                             col_name=column)
                elif method == "title-colname-col":
                    text, shuffleText = self.title_header_col(table[column],
                                                              col_name=column,
                                                              table_title=self.naming_dict[
                                                                  table_name])
                elif method == "title-colname-col-context":
                    text, shuffleText = self.title_header_col_context(table[column],
                                                                      col_name=column,
                                                                      table_title=self.naming_dict[
                                                                          table_name])
                elif method == "title-colname-stat-col":
                    text, shuffleText = self.title_header_stat_col(table[column],
                                                                   col_name=column,
                                                                   table_title=self.naming_dict[
                                                                       table_name])
                else:
                    raise ValueError(f"Method {method} not supported")

                table_column_representations[column] = text
                if self.shuffle:
                    shuffled_column_representations[column] = shuffleText
               # print( text,"\n shuffled", shuffleText,"\n" )
            column_representations[table_name] = table_column_representations
            if self.shuffle:
                shuffled_representations[table_name] = shuffled_column_representations
            n += 1
            """if n == 50:
                break"""
        return column_representations,shuffled_representations

    def col(self, column: pd.Series, prefix_text: str = '',
            suffix_text: str = ''):
        def token(cell):
            if isinstance(cell, str):
                return tokenize_with_number(cell)
            else:
                return cell

        column = column.dropna()
        column = column.apply(token)
        value_counts = column.value_counts()
        column_dict = value_counts.to_dict()
        transformed_col,shuffled_one = self._concat_until_max_length(column_dict, prefix_text,
                                                        suffix_text)
        return transformed_col, shuffled_one

    def _concat_until_max_length(self, column_dict: dict, initial_transformed_col: str = '',
                                 suffix_to_transformed_col: str = ''):
        def concate(list_constent):
            transformed_col = initial_transformed_col
            for index, cell in enumerate(list_constent):
                new_text_without_suffix = transformed_col + ', ' + str(
                    cell) if index > 0 else transformed_col + ' ' + str(
                    cell)
                new_text = new_text_without_suffix + suffix_to_transformed_col
                token_count = len(self.tokenizer(new_text))
                if token_count > self.max_length:
                    break
                transformed_col = new_text_without_suffix
            return transformed_col

        content = list(column_dict.keys())
        if self.shuffle:
            shuffle_list = content.copy()
            random.shuffle(shuffle_list)
            return concate(content),concate(shuffle_list)
        else:
            return concate(content), ""

    def header_col(self, column: pd.Series, col_name: str):
        transformed_col = col_name + ": "
        return self.col(column, transformed_col)

    def header_col_context(self, column: pd.Series, col_name: str, context: pd.DataFrame):
        prefix_text = col_name + ": "
        suffix_text = ". " + context
        return self.col(column, prefix_text, suffix_text) + suffix_text

    def header_stat_col(self, column: pd.Series, col_name: str):
        stats = _calculate_column_statistics(column)
        prefix_text = f"{col_name} contains {stats[0]} values ({stats[1]}, {stats[2]}, {stats[3]:.2f}): "
        return self.col(column, prefix_text)

    def title_header_col(self, column: pd.Series, col_name: str, table_title: str):
        prefix_text = f"{table_title}. {col_name}: "
        return self.col(column, prefix_text)

    def title_header_col_context(self, column: pd.Series, col_name: str, table_title: str,
                                 context: str = None):
        prefix_text = f"{table_title}. {col_name}: "
        suffix_text = ". " + context
        col_text, shuffled = self.col(column, prefix_text, suffix_text)
        return col_text + suffix_text, shuffled + suffix_text

    def title_header_stat_col(self, column: pd.Series, col_name: str, table_title: str):
        stats = _calculate_column_statistics(column)
        prefix_text = f"{table_title}. {col_name} contains {stats[0]} values ({stats[1]}, {stats[2]}, {stats[3]:.2f}): "
        return self.col(column, prefix_text)
