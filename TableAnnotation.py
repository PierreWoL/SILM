import pandas as pd
import SubjectColumnDetection as SCD
# import webSearchAPI as Search
import d3l.utils.functions as util
import numpy as np
import math


# from typing import Iterable


class TableColumnAnnotation:
    def __init__(self, table: pd.DataFrame, isCombine=False):

        if isinstance(table, pd.DataFrame) is False:
            print("input should be dataframe!")
            return
        self.table = table.copy(True)
        self.annotation = {}
        self.NE_table = table.copy(True)
        for i, col in enumerate(self.NE_table.columns):
            self.NE_table.rename(columns={col: str(i)}, inplace=True)
        # print(self.table)
        self.NE_cols = []
        self.annotate_type()
        # in the NE_columns() self.NE_table is also updated to store the tokens of cells in named_entity columns
        self.NE_columns()
        self.vocabulary = self.vocabularySet()
        self.matrix_list = []
        self.subject_col = []

    def annotate_type(self):
        """
        Preprocessing part in the TableMiner+ system
        classifying the cells from each column into one of the types mentioned in
        the ColumnType
        Returns -- self.annotation
        dictionary {header: column type}
        -------
        """
        # self.table.shape[1] is the length of columns
        for i in range(self.table.shape[1]):
            column = self.table.iloc[:, i]
            column_detection = SCD.ColumnDetection(column)
            candidate_type = column_detection.column_type_judge(100)
            self.annotation[i] = candidate_type
        """
        # since a lot of tables have no headers so that i use iloc
        for header in self.table.columns:
            column_detection = SCD.ColumnDetection(self.table[header])
            candidate_type = column_detection.column_type_judge(3)
            self.annotation[header] = candidate_type
        """
        # print(self.annotation)

    def NE_columns(self):
        """
        Returns the dictionary recording the NE columns header and its index in the table
        -------
        """
        for key, value in self.annotation.items():
            if value == SCD.ColumnType.named_entity:
                self.NE_cols.append(key)
        self.NE_table = self.NE_table.iloc[:, self.NE_cols]
        self.NE_table = self.NE_table.map(convert_to_tuple)  # + ":[]" #applymap

        """
        NE_Table should be like this:
        index NE_Column1                     NE_column2     NE_column3
        0       ([as,dc,],ws_score, bowTij )    ...            ...
        1        ...                            ...            ...
        ....        
        """

    def vocabularySet(self):
        vocabulary_set = set()
        for index, row in self.NE_table.iterrows():
            for cell in row:
                vocabulary_set.update(cell[0])
        self.vocabulary = list(vocabulary_set)
        return self.vocabulary

    """
            todo: fetch result but still in the progress
            calculate the context match score:
            TableMiner+ explanation: the frequency of the column header's composing words in the header's context
            Returns
            -------
            none but update the results self.ws
            """
    """def ws_cal(self, top_n: int, cseid="87f5671ca9e2242a9"):
        
        if self.NE_cols:
            self.NE_table = I_inf(self.table, self.NE_table, self.ws_table_cal, have_converged, top_K=top_n,
                                  cse_id=cseid)

        # print(self.NE_table)
        for row in self.table.iterrows():
            self.ws_table_cal(row, top_n)"""

    """
          this function calculates the ws score of each cell in the
          row

          Parameters
          ----------
          cse_id
          pairs : key value pairs
          row: row in table.iterrows()
          top_K: number of top searched web pages

          Returns
          -------
          should return or update the cells' ws score
          """
    """def ws_table_cal(self, row: tuple, pairs: pd.DataFrame, top_K=3, cse_id="87f5671ca9e2242a9"):
        # TODO  you forget about the returning value
        # in this case: pairs is the self.NE_table
      
        # series of all named_entity columns, the index of named entity columns
        # is obtained when detecting the type of table
        # row[0]: the index of the row of cell
        row_NE_cells = pd.Series([str(row[1][i]) for i in self.NE_cols])
        # concatenating all the named entity cells in the row as an input query
        input_query = row_NE_cells.str.cat(sep=",")
        # results are the returning results in dictionary format of top n web pages
        # P (webpages) in the paper
        results = Search.WebSearch(cse_id).search_result(input_query, top_K)
        # I_inf(tableAnnotated.table, ws_table_cal, convergence_pair, index=index_ne, top_K=top_n)
        # the D is collections of data rows Ti, and key value pair as <Tj, ws(Tj)>
        list_row_NE = list(row_NE_cells)
        for cell in list_row_NE:
            # column_index is the index of column in NE_table (correspondingly)
            # print(list_row_NE, cell)
            column_index = list_row_NE.index(cell)
            bow = self.bow_column(row[0], column_index)
            ws = ws_Tij(cell, results, bow)
            # print(pairs.iloc[row[0]][column_index][1])
            pairs.iloc[row[0]][column_index][1] = ws
            # print(pairs.iloc[row[0]][column_index][1], pairs.iloc[row[0] - 1][column_index][1])"""

    def bow_column(self, row_index, column_index) -> np.array:
        """
        Calculate the bow of each cell in the columns
        Parameters
        column_index: Note this is not the index of column in the
        original table but in the NE_table
        index: the index of cell in the row
        ----------
        Returns
        -------

        """
        bow = [0] * len(self.vocabulary)
        for token in self.NE_table.iloc[row_index, column_index][0]:
            bow[self.vocabulary.index(token)] += 1
        return np.array(bow)

    def columns_feature(self):
        """
        calculate all features of the columns
        Returns
        -------

        """
        for i in range(self.table.shape[1]):
            columnDetection = SCD.ColumnDetection(self.table.iloc[:, i])
            feature = columnDetection.features(i, self.NE_table)
            self.matrix_list.append(feature)
        matrix = np.array(self.matrix_list)
        # print(matrix)
        return matrix

    def subCol(self, threshold):
        score = subCol_Tj(self.columns_feature())
        score_l = list(score)
        # print(self.matrix_list, score_l,self.annotation)
        subject_column = [0] * len(score)
        score_l.sort(reverse=True)
        index0 = list(score).index(score_l[0])
        subject_column[index0] = 1
        if score_l[0] - score_l[1] <= threshold:
            index1 = list(score).index(score_l[1])
            subject_column[index1] = 1
        # print(subject_column)
        self.subject_col = subject_column
        return np.array(subject_column)


def subCol_Tj(mat: np.array):
    matrix = np.transpose(mat)
    features, df = matrix[:-1, :], matrix[-1, :]
    # Get the minimum and maximum values of each row
    row_min = np.min(features, axis=1).reshape(-1, 1)
    row_max = np.max(features, axis=1).reshape(-1, 1)

    # Subtract the minimum value of each row from all elements in the row
    # and Divide the result by the range (max - min) of each row
    def divide_array(dividend, divisor):
        for i, d in enumerate(divisor):
            if divisor[i] == [0]:
                divisor[i] = 1
        results = dividend / divisor
        return results

    # print(features)
    features = divide_array(features - row_min, row_max - row_min)
    normal_f = np.transpose(np.vstack((features, df)))
    """
    subcol(Tj) = (ucnorm(Tj) +2 *(cmnorm(Tj)+wsnorm(Tj)-emcnorm(Tj)) )/(df(Tj)+1)^(1/2)
    """

    def calc_func(row):
        return (row[1] + 2 * (row[2] + row[3]) - row[4]) / math.sqrt(row[5] + 1)

    # Use np.apply_along_axis to apply the calculation function to each row of the array
    return np.apply_along_axis(calc_func, 1, normal_f)


def ws_Tij(cell_text: str, webpages: list, bow_cell: np.array):
    """

    Parameters
    ----------
    cell_text
    webpages
    bow_cell

    Returns
    -------
    NOTE: freq_title is a list [], which length is
    """
    cell_token = util.remove_stopword(cell_text)
    if len(cell_token) == 0:
        cell_token.append(cell_text)
    length = len(cell_token)
    if len(cell_token) > 1 or len(cell_token) == 0:
        length += 1
    freq_title, freq_snippet = [0] * length, [0] * length
    for webpage in webpages:
        # todo need to check if there is no return, freq still the same after the function runs
        freq_count_web_page(cell_token, webpage, freq_title, freq_snippet)
    # print(freq_title, freq_snippet)
    countP = freq_title[0] * 2 + freq_snippet[0]
    countW = countw(freq_title, freq_snippet, bow_cell)
    return countP + countW


def freq_count_web_page(cell_token: list, webpage: dict, freq_title: list, freq_snippet: list):
    # the frequency of cell in title and snippet
    # print(cell_token, freq_title, freq_snippet)
    title, snippet = "", ""
    if "title" in webpage.keys():
        title = webpage["title"]
    if "snippet" in webpage.keys():
        snippet = webpage["snippet"]
    freq_title[0] += title.count(cell_token[0])
    freq_snippet[0] += snippet.count(cell_token[0])
    if len(cell_token) > 1:
        for index, token in enumerate(cell_token, 1):
            # print(index, token, len(freq_title), len(freq_snippet))
            freq_title[index] += util.tokenize_str(title).count(token)
            freq_snippet[index] += util.tokenize_str(snippet).count(token)


def countw(freq_title: list, freq_snippet: list, bowTij: np.array):
    countW = 0
    magnitudeW = np.linalg.norm(bowTij)
    if len(freq_title) > 1:
        for i in range(1, len(freq_title)):
            countW += freq_title[i] * 2 + freq_snippet[i]
        return countW / magnitudeW
    return countW


def convert_to_tuple(cell):
    cell_token = util.token_stop_word(cell)
    if util.token_stop_word(cell) != ['']:
        cell_token = [cell]
    return [cell_token, 0]


def I_inf(dataset,
          pairs: pd.DataFrame,
          process,
          convergence,
          **kwargs):
    """
    Implement the i-inf algorithm in the TableMiner+
    Parameters
    ----------
    pairs: collection of <key, value> pairs
    dataset : D in the paper representing datasets
    process : a function that processes data in the dataset and return the
    updated <key, value> pair
    update : insert the new key, value pair into the key,value pair list
    convergence: judge between the <key,value> pair i-1
    and <key, value> if the list has convergence

    Returns
    -------
    the collections of key value pairs ranked by v
    """
    i = 0
    iter_length = len(dataset)
    if type(dataset) is pd.DataFrame:
        iter_length = dataset.shape[0]
    for i in range(iter_length - 1):
        if i == 0:
            process((i, dataset.iloc[i]), pairs, **kwargs)

        i += 1
        process((i, dataset.iloc[i]), pairs, **kwargs)
        # if key in the pairs, update this key value pair, if not, add into key value pairs list
        if convergence(pairs.iloc[i - 1,], pairs.iloc[i,]):  #
            pairs = pairs.drop(index=pairs.index[i + 1:])
            break
    print(pairs)
    return pairs


def entropy(inputs):
    # key_value_pairs = {key: value for key, value in list(inputs)}
    key_value_pairs = {str(t[0]): t[1] for t in list(inputs)}
    entropy_cal = 0

    total = sum(key_value_pairs.values())
    for value in key_value_pairs.values():
        if value == 0:
            continue
        entropy_cal -= value / total * math.log(value / total)
    return entropy_cal


def have_converged(pre_pairs, cur_pairs, threshold=0.1):
    previous_entropy = entropy(pre_pairs)
    current_entropy = entropy(cur_pairs)
    if abs(current_entropy - previous_entropy) < threshold:
        return True
    return False
