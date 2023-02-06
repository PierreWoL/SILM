import math
import re
import pandas as pd
from typing import Iterable
from d3l.input_output.dataloaders import CSVDataLoader
import d3l.utils.functions as func
import os
import datetime
from typing import Iterator, List, Optional, Tuple, Union, Dict, Any
from enum import Enum
import webSearchAPI as search


class ColumnType(Enum):
    Invalid = -1
    long_text = 0
    named_entity = 1
    number = 2
    date_expression = 3
    empty = 4
    other = 5


class TableColumnAnnotation:
    def __init__(self, table: pd.DataFrame):

        if isinstance(table, pd.DataFrame):
            print("input should be dataframe!")
            pass
        self.table = table
        self.ws = 0
        self.annotation = {}
        self.NE_cols = {}

    #   todo TBC

    def annotate_type(self):
        """
        Preprocessing part in the TableMiner+ system
        classifying the cells from each column into one of the types mentioned in
        the ColumnType
        Returns -- self.annotation
        dictionary {header: column type}
        -------

        """
        for header in self.table.columns:
            candidate_type = ColumnDetection(self.table[header]).col_type
            self.annotation[header] = candidate_type

    def NE_columns(self) -> dict:
        """
        Returns the dictionary recording the NE columns header and its index in the table
        -------

        """
        key_list = list(self.annotation.keys())
        for key, value in self.annotation:
            if value == ColumnType.named_entity:
                key_list.index(key)
                self.NE_cols[key] = key_list.index(key)
        return self.NE_cols

    def ws_cal(self, table: pd.DataFrame, top_n: int):
        """
        todo: fetch result but still in the progress
        calculate the context match score:
        TableMiner+ explanation: the frequency of the column header's composing words in the header's context
        Returns
        -------
        """
        for index, row in self.table.iterrows():
            row_temp = table.iloc[index]
            input_query = row_temp.astype(str).str.cat(sep=" ")
            results = search.WebSearch().search_result(input_query, top_n)
            for cell in row:
                cell_freq_title = 0
                if func.has_numbers(cell) is False:
                    for result in results:
                        cell_freq_title = cell_freq_title + result["title"].count(cell)
                        print(cell_freq_title, result["title"].count(cell))


class ColumnDetection:

    def __init__(self, values: Iterable[Any]):
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        self.column = values
        self.col_type = ColumnType.Invalid
        '''
        feature used in the subject column detection
        emc: fraction of empty cells
        uc: fraction of cells with unique content 
        ac: if over 50% cells contain acronym or id
        df: distance from the first NE-column
        cm: context match score
        ws: web search score
        '''
        self.emc = 0
        self.uc = 0
        self.ac = 0
        self.df = 0
        self.cm = 0
        self.acronym_id_num = 0

    def column_type_judge(self):
        """
        Check the type of given column's data type.

        Parameters
        NOTE: I add the tokenize the column step in this function,
        maybe in the future I need to extract it as a independent function
        ----------
        values :  Iterable[Any] A collection of values.
        Returns
        -------
        bool
           All non-null values are long text or not (True/False).
        """
        col_type = -1
        type_count = [0, 0, 0, 0, 0, 0]
        total_token_number = 0
        temp_count_text_cell = 0
        try:
            if not self.column:
                raise ValueError("Column does not exist!")
        except ValueError as e:
            print("function terminate.", repr(e))

        # iterate and judge the element belong to which category
        for index, element in self.column.items():
            if index == math.floor(len(self.column) / 3):
                if temp_count_text_cell != 0:
                    ave_token_number = total_token_number / temp_count_text_cell
                    # TODO : I think this needs further modification later Currently set to 10 just in case
                    if ave_token_number > 10:
                        type_count[ColumnType.long_text.value] = temp_count_text_cell
                    else:
                        type_count[ColumnType.named_entity.value] = type_count[ColumnType.named_entity.value] + \
                                                                    temp_count_text_cell
                print(type_count)
                col_type = type_count.index(max(type_count))
                break
            print(element)
            # if it is int type
            if isinstance(element, int):
                type_count[ColumnType.number.value] = type_count[ColumnType.number.value] + 1
                continue
            if func.is_empty(element):  # or pd.isna(element)
                type_count[ColumnType.empty.value] = type_count[ColumnType.empty.value] + 1
                continue
            else:
                # judge string type
                # print(token, token_with_number)
                if len(element.split(" ")) == 1:
                    # Judge if it is a null value
                    # TODO : need to mark this empty cell and calculate how many empty cells exist
                    remove_punc_ele = element.translate(str.maketrans('', '', ','))
                    # Judge if it is a numeric value
                    if func.is_number(element):
                        # There exists special cases: where year could be recognized as number
                        if element.isdigit() & int(element) >= 1000 & \
                                int(element) <= int(datetime.datetime.today().year):
                            type_count[ColumnType.date_expression.value] = type_count[
                                                                               ColumnType.date_expression.value] + 1
                        else:
                            type_count[ColumnType.number.value] = type_count[ColumnType.number.value] + 1
                        continue
                    if func.is_number(remove_punc_ele):
                        type_count[ColumnType.number.value] = type_count[ColumnType.number.value] + 1
                        continue

                    # IF non-numeric, judge if it is a date-expression
                    if func.is_date_expression(element):
                        type_count[ColumnType.date_expression.value] = type_count[ColumnType.date_expression.value] + 1
                        continue

                    # judge if it is a single word indicate an entity or a acronym
                    if element.isalpha():
                        # actually this is an acronym type, but this will fixed in the future todo: later add this type
                        if func.is_acronym(element):
                            type_count[ColumnType.other.value] = type_count[ColumnType.other.value] + 1
                            continue
                        else:
                            type_count[ColumnType.named_entity.value] = type_count[ColumnType.named_entity.value] + 1
                            continue
                    else:
                        # maybe id type cell
                        type_count[ColumnType.other.value] = type_count[ColumnType.other.value] + 1
                        continue
                else:
                    token_str = func.tokenize(element)
                    token = token_str.split(" ")
                    token_with_number = func.tokenize_with_number(element).split(" ")
                    if len(token_with_number) == 2:
                        if func.is_number(token_with_number[0]):
                            type_count[ColumnType.number.value] = type_count[ColumnType.number.value] + 1
                            continue
                        if func.is_date_expression(func.tokenize_with_number(element)):
                            type_count[ColumnType.date_expression.value] = type_count[
                                                                               ColumnType.date_expression.value] + 1
                            continue
                    if len(token) < 3:
                        acronym = True
                        for i in token:
                            if func.is_acronym(i) is False:
                                acronym = False
                                break
                        if acronym is True:
                            type_count[ColumnType.other.value] = type_count[ColumnType.other.value] + 1
                    else:
                        total_token_number = total_token_number + len(token)
                        temp_count_text_cell = temp_count_text_cell + 1

            # stop iteration to the 1/3rd cell and judge what type occupies the most in columns
        self.acronym_id_num = type_count[ColumnType.other.value]
        self.col_type = ColumnType(col_type).name
        # return col_type

    def emc_cal(self):
        """
        Calculate the fraction of empty cells
        Returns none
        -------
        """
        empty_cell_count = 0
        for ele in self.column:
            if func.is_empty(ele):
                empty_cell_count = empty_cell_count + 1
        self.emc = empty_cell_count / len(self.column)

    def uc_cal(self):
        """
              Calculate the fraction of cells with unique text
              a ratio between the number of unique text content and the number of rows
              Returns none
              -------
        """
        column_tmp = self.column
        column_tmp.drop_duplicates()
        self.uc = len(column_tmp) / len(self.column)

    def ac_cal(self):
        """
            indicate if more than 50% cells of a column is acronym
            or id
            -------
        """
        if self.acronym_id_num / len(self.column) > 0.5:
            self.ac = 1

    # todo need to first finish the table annotation class then  we can help

    def df_cal(self, index, annotation_table: TableColumnAnnotation):

        if self.column.name not in annotation_table.NE_cols.keys():
            print("Error! This column is not in this table!")
            pass
        else:
            first_pair = next(iter((annotation_table.NE_cols.items())))
            self.df = annotation_table.NE_cols.get(self.column.name) - first_pair[1]
            """
            temp_annotation = list(annotation_table.annotation.items())
            # pass the index of this column through parameter
            res = [index for index, pair in enumerate(temp_annotation) if pair[0] == self.col_type]
            """

    def cm_cal(self):
        """
        calculate the context match score:
        TableMiner+ explanation: the frequency of the column header's composing words in the header's context
        Returns
        -------
        """


# def I_inf(dataset,):

data_path = os.getcwd() + "/T2DV2/test/"
# print(subject.tables[0].iloc[:,0])
example = '/Users/user/My Drive/CurrentDataset/T2DV2/test/3887681_0_7938589465814037992.csv'
tableExample = pd.read_csv(example)
detection = ColumnDetection(tableExample.iloc[:, 4])
typeTest = detection.column_type_judge()
# print(columnType(typeTest).name)
print(detection.col_type)

'''
This is random test for the column-detection
TODO: write a test function that can randomly choose table 's column and 
detect its type
if type is invalid(-1) throw exception
'''

# print(func.is_long_text(tableExample.iloc[:,2]))
'''
def column_type_detection(table):
    for i in table.columns:
        for element in table[i]:
'''

'''
Used in read tables, Useless now but may be a little helpful
in the future

    def __init__(self, root_path: str, **loading_kwargs: Any):
        super().__init__(root_path, **loading_kwargs)
        if not os.path.isdir(root_path):
            raise FileNotFoundError(
                "The {} root directory was not found locally. "
                "A CSV loader must have an existing directory associated!".format(
                    root_path
                )
            )
        self.data_path = root_path
        if self.data_path[-1] != "/":
            self.data_path = self.data_path + "/"
        self.loading_kwargs = loading_kwargs
        self.data_path = root_path
        self.tables = []
        CSVDataLoader(root_path=(root_path), encoding='latin-1')

    def read_tables(self):
        T = os.listdir(self.data_path)
        T = [t[:-4] for t in T if t.endswith('.csv')]
        T.sort()
        dataloader = CSVDataLoader(
            root_path=(self.data_path),
            encoding='latin-1'
        )
        for t in T:
            table = dataloader.read_table(table_name=t)
            # print(table)
            self.tables.append(table)

    def Tables(self):
        return self.tables
'''

# print(subject.tables[0].columns.tolist())
