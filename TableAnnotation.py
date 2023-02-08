import pandas as pd
import SubjectColumnDetection as SCD
import webSearchAPI as Search
import d3l.utils.functions as util
import numpy as np
import math


class TableColumnAnnotation:
    def __init__(self, table: pd.DataFrame):

        if isinstance(table, pd.DataFrame):
            print("input should be dataframe!")
            pass
        self.table = table
        self.annotation = {}
        self.NE_table = table
        self.NE_cols = {}
        # in the NE_columns() self.NE_table is also updated to store the tokens of cells in named_entity columns
        self.NE_cols = self.NE_columns()
        self.vocabulary = self.vocabularySet()

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
            candidate_type = SCD.ColumnDetection(self.table[header]).col_type
            self.annotation[header] = candidate_type

    def NE_columns(self) -> dict:
        """
        Returns the dictionary recording the NE columns header and its index in the table
        -------

        """
        key_list = list(self.annotation.keys())
        for key, value in self.annotation:
            if value == SCD.ColumnType.named_entity:
                key_list.index(key)
                self.NE_cols[key] = key_list.index(key)

        self.NE_table = self.table[list(self.NE_cols.keys())]
        self.NE_table = self.NE_table.applymap(add_colon_zero)  # + ":[]"
        """
        NE_Table should be like this:
        index NE_Column1                     NE_column2     NE_column3
        0       ([as,dc,],ws_score, bowTij )    ...            ...
        1        ...                            ...            ...
        ....        
        """
        return self.NE_cols

    def vocabularySet(self):
        vocabulary_set = set()
        for index, row in self.NE_table.iterrows():
            for cell in row:
                vocabulary_set.update(util.token_stop_word(cell))
        self.vocabulary = list(vocabulary_set)
        return self.vocabulary

    def ws_cal(self, top_n: int):
        """
        todo: fetch result but still in the progress
        calculate the context match score:
        TableMiner+ explanation: the frequency of the column header's composing words in the header's context
        Returns
        -------
        none but update the results self.ws
        """
        index_ne = list(self.NE_cols.values())
        I_inf(self.table, self.NE_table, self.ws_table_cal, convergence_pair, top_K=top_n)

        # for row in self.table.iterrows():
        #    self.ws_table_cal(row, top_n)

    def ws_table_cal(self, row: tuple, top_K: int, pairs: pd.DataFrame):
        # TODO  you forget about the returning value
        # in this case: pairs is the self.NE_table
        """
        this function calculates the ws score of each cell in the
        row

        Parameters
        ----------
        row: row in table.iterrows()
        top_K: number of top searched web pages

        Returns
        -------
        should return or update the cells' ws score
        """
        # series of all named_entity columns, the index of named entity columns
        # is obtained when detecting the type of table
        # row[0]: the index of the row of cell
        row_NE_cells = pd.Series([row[1][i] for i in list(self.NE_cols.values())])
        # concatenating all the named entity cells in the row as an input query
        input_query = row_NE_cells.str.cat(sep=",")
        # results are the returning results in dictionary format of top n web pages
        # P (webpages) in the paper
        results = Search.WebSearch().search_result(input_query, top_K)
        # I_inf(tableAnnotated.table, ws_table_cal, convergence_pair, index=index_ne, top_K=top_n)
        # the D is collections of data rows Ti, and key value pair as <Tj, ws(Tj)>
        for cell in row_NE_cells:
            # column_index is the index of column in NE_table (correspondingly)
            column_index = row_NE_cells.index(cell)
            print(column_index)
            bow = self.bow_column(row[0], column_index)
            ws = ws_Tij(cell, results, bow)
            # 在这里我先算每个的 不累加
            pairs.iloc[row[0]][column_index][1] = ws
            print(pairs.iloc[row[0]][column_index][1], pairs.iloc[row[0] - 1][column_index][1])

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
        for token in self.NE_table.iloc[row_index, column_index]:
            bow[self.vocabulary.index(token)] += 1
        return np.array(bow)


def ws_Tij(cell_text: str, webpages: list, bow_cell: np.array):
    cell_token = util.remove_stopword(cell_text)
    length = len(cell_token)
    freq_title, freq_snippet = [] * (length + 1), [] * (length + 1)
    for webpage in webpages:
        # todo need to check if there is no return, freq still the same after the function runs
        freq_count_web_page(cell_token, webpage, freq_title, freq_snippet)
    countP = freq_title[0] * 2 + freq_snippet[0]
    countW = countw(freq_title, freq_snippet, bow_cell)
    return countP + countW


def freq_count_web_page(cell_token: list, webpage: dict, freq_title: list, freq_snippet: list):
    # the frequency of cell in title and snippet
    freq_title[0] += webpage["title"].count(cell_token[0])
    freq_snippet[0] += webpage["snippet"].count(cell_token[0])
    if len(cell_token) == 2:
        freq_title.pop()
        freq_snippet.pop()
    else:
        for index, token in enumerate(cell_token, 1):
            freq_title[index] += util.tokenize_str(webpage["title"]).count(token)
            freq_snippet[index] += util.tokenize_str(webpage["snippet"]).count(token)


def countw(freq_title: list, freq_snippet: list, bowTij: np.array):
    sumW = 0
    magnitudeW = np.linalg.norm(bowTij)
    if len(freq_title) > 1:
        for i in range(1, len(freq_title)):
            sumW += freq_title[i] * 2 + freq_snippet[i]
    return sumW / magnitudeW


def convergence_pair(pair1: tuple, pair2: tuple):
    # todo CARRY ON HERE
    return 1


def add_colon_zero(cell):
    return "{}: 0".format(cell)


def I_inf(dataset,
          pairs: pd.DataFrame,
          process,
          convergence,
          **kwargs) -> list:
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
    iters = dataset
    if type(dataset) is pd.DataFrame:
        iters = dataset.iterrows()
    for data in iters:
        if i == 0:
            process(data, pairs, **kwargs)
            continue
        i += 1
        process(data, pairs, **kwargs)
        # if key in the pairs, update this key value pair, if not, add into key value pairs list
        if convergence(pairs.iloc[i - 1, ], pairs.iloc[i, ]):
            break
        pairs.drop(index=pairs.index[i+1:])
    return pairs


def entropy(key_value_pairs):
    entropy_cal = 0
    total = sum(key_value_pairs.values())
    for value in key_value_pairs.values():
        entropy_cal -= value / total * math.log(value / total)
    return entropy_cal


def have_converged(pre_pairs, cur_pairs, threshold=0.0001):
    previous_entropy = entropy(pre_pairs)
    current_entropy = entropy(cur_pairs)
    if abs(current_entropy - previous_entropy) < threshold:
        return True
    return False
