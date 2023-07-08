import math
import string

import pandas as pd
from typing import Iterable
from d3l.input_output.dataloaders import CSVDataLoader
import d3l.utils.functions as func
import os
import datetime
from typing import Any
from enum import Enum
import random
import statistics
from d3l.utils.constants import STOPWORDS
from nltk.stem import WordNetLemmatizer
import experimentalData as ed
from country_list import countries_for_language
"""
This combines the features to detection subject columns of tables
both in tableMiner+ and recovering semantics of data on the web
1. column_type_judge: judge the type of column:
    named_entity, long_text, number, date_expression, empty, other
"""


def is_country(string):
    token = func.tokenize_str(string)
    countries = [x[1] for x in countries_for_language('en')]


class ColumnType(Enum):
    Invalid = -1
    long_text = 0
    named_entity = 1
    number = 2
    date_expression = 3
    empty = 4
    other = 5


class ColumnDetection:
    def __init__(self, values: Iterable[Any]):

        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        self.column = values
        self.col_type = self.column_type_judge(3)

        '''
        feature used in the subject column detection
       
        emc: fraction of empty cells
        uc: fraction of cells with unique content 
        ac: if over 50% cells contain acronym or id
        df: distance from the first NE-column
        cm: context match score (doubt this could work)
        ws: web search score
        
        additional ones in the recovering semantics of data 
        on the web
        
        tlc: average number of words in each cell
        vt: variance in the number of data tokens in each cell
        '''
        self.emc = 0
        self.uc = 0
        self.ac = 0
        self.df = 0
        self.cm = 0
        self.tlc = 0
        self.vt = 0
        self.ws = 0
        self.acronym_id_num = 0

    def column_type_judge(self, fraction=200):
        """
        Check the type of given column's data type.

        Parameters
        NOTE: I add the tokenize the column step in this function,
        maybe in the future I need to extract it as an independent function
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
            if len(self.column) == 0:
                raise ValueError("Column does not exist!")
        except ValueError as e:
            print("column_type_judge terminate.", self.column.name, repr(e))
            pass
        checkpoint = fraction
        # iterate and judge the element belong to which category
        for index, element in self.column.items():

            if checkpoint>= len(self.column):
                checkpoint = len(self.column)-1
            if index == checkpoint:
                if temp_count_text_cell != 0:
                    ave_token_number = total_token_number / temp_count_text_cell
                    # TODO : I think this needs further modification later Currently set to 10 just in case
                    if ave_token_number > 7:
                        type_count[ColumnType.long_text.value] = temp_count_text_cell
                    else:
                        type_count[ColumnType.named_entity.value] = type_count[ColumnType.named_entity.value] + \
                                                                    temp_count_text_cell
                # print(type_count)
                col_type = type_count.index(max(type_count))
                break
            # print(element, type(element))
            # if this cell is empty
            if func.is_empty(element):
                type_count[ColumnType.empty.value] += 1
                continue
            # if it is int/float type
            if isinstance(element, int) or isinstance(element, float):
                if isinstance(element, int) and 1000 <= element <= int(datetime.datetime.today().year):
                    type_count[ColumnType.date_expression.value] += 1
                    continue
                if func.is_empty(str(element)):
                    type_count[ColumnType.empty.value] += 1
                    continue
                else:
                    type_count[ColumnType.number.value] += 1
                    continue

            else:
                # judge string type
                # print(token, token_with_number)
                if len(element.split(" ")) == 1:
                    # Judge if it is a null value
                    # TODO : need to mark this empty cell and calculate how many empty cells exist
                    if func.is_number(element):
                        # There exists special cases: where year could be recognized as number
                        type_count[ColumnType.number.value] += 1
                        continue
                    # Judge if it is a numeric value
                    if ',' in element:
                        remove_punc_ele = element.translate(str.maketrans('', '', ','))
                        if func.is_number(remove_punc_ele):
                            type_count[ColumnType.number.value] += 1
                            continue
                    # IF non-numeric, judge if it is a date-expression
                    if func.is_date_expression(element):
                        type_count[ColumnType.date_expression.value] += 1
                        continue

                    # judge if it is a single word indicate an entity or a acronym

                    if element.isalpha():
                        # actually this is an acronym type, but this will fixed in the future todo: later add this type
                        if func.is_acronym(element):
                            if is_country(element) is True:
                                type_count[ColumnType.named_entity.value] += 1
                                continue
                            else:
                                type_count[ColumnType.other.value] += 1
                                continue
                        else:
                            ele = func.tokenize_str(element).lower()
                            lemmatizer = WordNetLemmatizer()
                            ele_origin = lemmatizer.lemmatize(ele)
                            if ele_origin not in STOPWORDS and ele_origin != 'yes' and ele_origin != 'no':
                                type_count[ColumnType.named_entity.value] += 1
                                continue

                            else:
                                type_count[ColumnType.other.value] += 1
                    if func.is_valid_url(element) is True:
                        type_count[ColumnType.other.value] += 1
                    else:
                        tokens = func.token_stop_word(element)
                        if func.is_acronym(element.translate(str.maketrans('', '', string.digits))) is True:
                            type_count[ColumnType.other.value] += 1
                            continue
                        judge = False
                        for token in tokens:
                            if token.isalpha() is True and func.is_acronym(token) is False:
                                judge = True
                                continue
                        if judge is True:
                            type_count[ColumnType.named_entity.value] += 1
                        else:
                            type_count[ColumnType.other.value] += 1
                else:
                    token_str = func.tokenize_str(element)
                    token = token_str.split(" ")
                    token_with_number = func.tokenize_with_number(element).split(" ")
                    if len(token_with_number) == 2:
                        if func.is_number(token_with_number[0]):
                            type_count[ColumnType.number.value] += 1
                            continue
                        if func.is_date_expression(func.tokenize_with_number(element)):
                            type_count[ColumnType.date_expression.value] += 1
                            continue
                    if len(token) < 3:
                        acronym = True
                        for i in token:
                            if func.is_acronym(i) is False:
                                acronym = False
                                break
                        if acronym is True:
                            type_count[ColumnType.other.value] += 1
                            continue
                        else:
                            type_count[ColumnType.named_entity.value] += 1
                            continue

                    else:
                        total_token_number = total_token_number + len(token)
                        temp_count_text_cell = temp_count_text_cell + 1

            # stop iteration to the 1/3rd cell and judge what type occupies the most in columns
        self.acronym_id_num = type_count[ColumnType.other.value]
        self.col_type = ColumnType(col_type)
        return self.col_type

    '''
        def column_token(self, annotation_table: Annotate.TableColumnAnnotation()):
        """
        return the column tokens that have already calculated in the annotated table
        Parameters
        ----------
        annotation_table: table that have already been annotated with type
        Returns
        -------

        """
        if annotation_table.NE_table == annotation_table.table:
            annotation_table.NE_columns()
        self.column_tokens = \
            {key: [0] * len(annotation_table.vocabularySet())
             for key in annotation_table.NE_table[self.column.name]}
    '''

    def emc_cal(self):
        """
        Calculate the fraction of empty cells
        Returns none
        -------
        """
        empty_cell_count = 0
        for ele in self.column:
            if func.is_empty(ele):
                empty_cell_count += 1
        self.emc = empty_cell_count / len(self.column)
        return self.emc

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

    def df_cal(self, index: int, NE_table: pd.DataFrame):
        """
        calculate the df score
        the distance between this NE column and the first NE column
        -------
         """
        if self.col_type == ColumnType.named_entity:
            first_pair = NE_table.columns[0]
            self.df = index - int(first_pair)
        """
            temp_annotation = list(annotation_table.annotation.items())
            # pass the index of this column through parameter
            res = [index for index, pair in enumerate(temp_annotation) if pair[0] == self.col_type]
        """

    def ws_cal(self, index: int, NE_table: pd.DataFrame):
        if self.col_type == ColumnType.named_entity:
            for item in NE_table[str(index)]:
                self.ws += item[1]

    def cm_cal(self):
        """
        calculate the context match score:
        TableMiner+ explanation: the frequency of the column header's composing words in the header's context
        note: In the paper they mention the different context include webpage title/table caption and surrounding
        paragraphs, currently our datasets doesn't include this, so we pass this score as 1
        Returns
        -------
        """
        if self.col_type != ColumnType.named_entity or self.col_type != ColumnType.long_text:
            # print("No need to calculate context match score!")
            pass
        else:
            self.cm = 1

    def tlc_cal(self):
        """
        variance in the number of data tokens in each cell
        Returns
        -------
        """
        if self.col_type == ColumnType.named_entity:
            token_list = list(self.column.apply((lambda x: len(func.token_stop_word(x)))))
            self.tlc = statistics.variance(token_list)
        # return self.tlc

    def vt_cal(self):
        self.vt = self.column.apply((lambda x: len(str(x).split(" ")))).sum() / len(self.column)
        # return self.vt

    def features(self, index: int, NE_table: pd.DataFrame):
        self.emc_cal()
        self.uc_cal()
        self.ac_cal()
        self.df_cal(index, NE_table)
        self.ws_cal(index, NE_table)
        self.cm_cal()
        self.vt_cal()
        self.tlc_cal()
        return [self.ac, self.uc, self.ws, self.cm, self.emc, self.df, self.vt, self.tlc]


def datasets(root_path):
    if not os.path.isdir(root_path):
        raise FileNotFoundError(
            "The {} root directory was not found locally. "
            "A CSV loader must have an existing directory associated!".format(
                root_path
            )
        )

    if root_path[-1] != "/":
        root_path = root_path + "/"
    tables = {}
    dataloader = CSVDataLoader(root_path=root_path, encoding='latin-1')
    T = os.listdir(root_path)
    T = [t[:-4] for t in T if t.endswith('.csv')]
    if len(T)>1:
        T.sort()
        #print(T)
        #random.choices(T, k=400)
    else:
        Table_names = []
        T = os.listdir(root_path)
        for t in T:
            if t!=".DS_Store" and not t.endswith(".csv"):
                for file in os.listdir(root_path+t+"/"):
                    if file.endswith('.csv'):
                        Table_names.append(t+"/"+file)
        T = Table_names.copy()
    for t in T:
        table = dataloader.read_table(table_name=t)
        #print(table.iloc[:,1])
        tables[t] = table
    return tables


def random_table(tables: dict):
    random_key = random.choice(list(tables.keys()))
    return random_key, tables[random_key]


def test_subject_column(filename):
    files = ed.get_files(filename)
    for file in files:
        f = open(filename + file + '.csv', errors='ignore')
        table = pd.read_csv(f)
        print(table.T)
        for i in range(0, table.shape[1]):
            annotation = ColumnDetection(table.iloc[:, i])
            print(annotation.column_type_judge(3))


'''
This is random test for the column-detection
TODO: write a test function that can randomly choose table 's column and 
detect its type
if type is invalid(-1) throw exception
'''

"""
example = '/Users/user/My Drive/CurrentDataset/T2DV2/test/3887681_0_7938589465814037992.csv'
tableExample = pd.read_csv(example)
detection = ColumnDetection(tableExample.iloc[:, 4])
typeTest = detection.column_type_judge(2)
print(detection.col_type)
"""

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
