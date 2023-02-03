import math
import string

import pandas as pd
from typing import Iterable
from d3l.input_output.dataloaders import CSVDataLoader
import d3l.utils.functions as func
import os
import datetime
import gensim
from typing import Iterator, List, Optional, Tuple, Union, Dict, Any
from enum import Enum
import regex
import fasttext

class columnType(Enum):
    long_text = 0
    named_entity = 1
    number = 2
    date_expression = 3
    empty = 4
    other = 5
'''
Used in read tables

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
#class tableColumnAnotate():
    #def __init__(self, table: pd.DataFrame)

class columnDetection():

    def __init__(self, values: Iterable[Any]):
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        self.column = values

    def column_type_judge(self) -> int:
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

            # stop iteration to the 1/3rd cell and judge what type occupies the most in columns
            if index == math.floor(len(self.column) / 3):
                if temp_count_text_cell != 0:
                    ave_token_number = total_token_number / temp_count_text_cell
                    # TODO : I think this needs further modification later Currently set to 10 just in case
                    if ave_token_number > 10:
                        type_count[columnType.long_text.value] = temp_count_text_cell
                    else:
                        type_count[columnType.named_entity.value] = type_count[columnType.named_entity.value] + \
                                                                    temp_count_text_cell
                print(type_count)
                col_type = type_count.index(max(type_count))
                break
            print(element)
            # if it is int type
            if isinstance(element,int):
                type_count[columnType.number.value] = type_count[columnType.number.value] + 1
                continue
            if func.is_empty(element) or pd.isna(element):
                type_count[columnType.empty.value] = type_count[columnType.empty.value] + 1
                continue
            else:
                # judge string type

                #print(token, token_with_number)
                if len(element.split(" ")) == 1:
                    # Judge if it is a null value
                    # TODO : need to mark this empty cell and calculate how many empty cells exist
                    remove_punc_ele = element.translate(str.maketrans('', '', ','))
                    # Judge if it is a numeric value
                    if func.is_number(element):
                        # There exists special cases: where year could be recognized as number
                        if element.isdigit() & int(element) >= 1000 & \
                                int(element) <= int(datetime.datetime.today().year):
                            type_count[columnType.date_expression.value] = type_count[
                                                                               columnType.date_expression.value] + 1
                        else:
                            type_count[columnType.number.value] = type_count[columnType.number.value] + 1
                        continue
                    if func.is_number(remove_punc_ele):
                        type_count[columnType.number.value] = type_count[columnType.number.value] + 1
                        continue

                    # IF non-numeric, judge if it is a date-expression
                    if func.is_date_expression(element):
                        type_count[columnType.date_expression.value] = type_count[columnType.date_expression.value] + 1
                        continue

                    # judge if it is a single word indicate an entity or a acronym
                    if element.isalpha():
                        # actually this is an acronym type, but this will fixed in the future todo: later add this type
                        if func.is_acronym(element):
                            type_count[columnType.other.value] = type_count[columnType.other.value] + 1
                            continue
                        else:
                            type_count[columnType.named_entity.value] = type_count[columnType.named_entity.value] + 1
                            continue
                    else:
                        type_count[columnType.other.value] = type_count[columnType.other.value] + 1
                else:
                    token_str = func.tokenize(element)
                    token = token_str.split(" ")
                    token_with_number = func.tokenize_with_number(element).split(" ")
                    if len(token_with_number) == 2:
                        if func.is_number(token_with_number[0]):
                            type_count[columnType.number.value] = type_count[columnType.number.value] + 1
                            continue
                        if func.is_date_expression(func.tokenize_with_number(element)):
                            type_count[columnType.date_expression.value] = type_count[columnType.date_expression.value] + 1
                            continue
                    if len(token) <3:
                        acronym = True
                        for i in token:
                            if func.is_acronym(i) is False:
                                acronym = False
                                break
                        if acronym is True:
                            type_count[columnType.other.value] = type_count[columnType.other.value] + 1
                    else:
                        total_token_number = total_token_number + len(token)
                        temp_count_text_cell = temp_count_text_cell + 1


        return col_type


data_path = os.getcwd() + "/T2DV2/test/"




# print(subject.tables[0].iloc[:,0])
example = '/Users/user/My Drive/CurrentDataset/T2DV2/test/2066887_0_746204117268168398.csv'
tableExample = pd.read_csv(example)
detection = columnDetection(tableExample.iloc[:, 1])
typeTest = detection.column_type_judge()
print(columnType(typeTest).name)



'''
This is random test for the column-detection
记得写测试函数 随机选取table i 和任意列
测试结果

如果是type = -1 through exception
'''

# print(func.is_long_text(tableExample.iloc[:,2]))
'''
def column_type_detection(table):
    for i in table.columns:
        for element in table[i]:
'''

# print(subject.tables[0].columns.tolist())
