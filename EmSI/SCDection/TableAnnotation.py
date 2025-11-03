from time import sleep

import pandas as pd
import SCDection.SubjectColumnDetection as SCD
import SCDection.webSearchAPI as Search
import math
import Utils as util


class TableColumnAnnotation:

    def __init__(self, table: pd.DataFrame, SearchingWeb=False):

        if isinstance(table, pd.DataFrame) is False:
            print("input should be dataframe!")
            return
        self.table = table.copy(True)
        self.annotation = {}
        self.annotate_type()
        self.subject_col = []
        self.search = Search.WebSearch()
        self.column_score = {}
        self.isWeb = SearchingWeb


    def annotate_type(self):
        """
        Preprocessing part in the TableMiner system
        classifying the cells from each column into one of the types mentioned in
        the ColumnType
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
            todo: fetch result but still in the progress
            calculate the context match score:
            TableMiner explanation: the frequency of the column header's composing words in the header's context
            Returns
            -------
            none but update the results self.ws
            """

    def ws_cal(self, top_n: int):
        ws_Ci = {}
        for i, candidate_type in self.annotation.items():
            if candidate_type == SCD.ColumnType.named_entity:
                # print(i, candidate_type)
                if self.isWeb is False:
                    self.column_score[i] = 0
                else:
                    ws_Ci_dict = util.I_inf(self.table.values.tolist(), ws_Ci, self.ws_cell_cal, self.update_ws,
                                            column_index=i, top_K=top_n)
                    self.column_score[i] = sum(ws_Ci_dict.values())

            # else:
            # self.column_score[i] = 0

    @staticmethod
    def update_ws(current_state, new_pairs,**kwargs):
        for cell, cell_score in new_pairs.items():
            current_state[cell] = cell_score
        return current_state

    def ws_cell_cal(self, row, index, column_index, top_K=3):
        # in this case: pairs is the self.NE_table
        # series of all named_entity columns, the index of named entity columns
        # is obtained when detecting the type of table
        # concatenating all the named entity cells in the row as an input query
        input_query = ' '.join([str(i) for i in row])

        # results are the returning results in dictionary format of top n web pages
        # P (webpages) in the paper
        results = self.search.search_result(input_query, top_K)
        sleep(2)
        cell = self.table.iloc[index, column_index]
        if isinstance(cell,float):
            cell_ws_score = 0
        else:
            cell_ws_score = self.countp(cell, results) + self.countw(cell, results)
        return {cell: cell_ws_score}

    @staticmethod
    def frequency_a_in_b(str_a, str_b):
        token_a = util.tokenize_with_number(str_a)
        token_b = util.tokenize_with_number(str_b)
        count = token_b.count(token_a)
        return count

    def countp(self, cell, webpages: dict):
        # the frequency of cell in title and snippet
        countp_score = 0
        for webpage in webpages:
            title_score = self.frequency_a_in_b(cell, webpage["title"])
            snippet_score = self.frequency_a_in_b(cell, webpage["snippet"])
            countp_score += 2 * title_score + snippet_score
        return countp_score

    def countw(self, cell, webpages: dict):
        countw_score = 0
        cell_token = util.bow(cell)
        len_bow_Tj = sum(cell_token.values())
        for webpage in webpages:
            for token in cell_token.keys():
                title_score = self.frequency_a_in_b(token, webpage["title"])
                snippet_score = self.frequency_a_in_b(token, webpage["snippet"])
                countw_score += (2 * title_score + snippet_score) / len_bow_Tj
        return countw_score

    def subcol_Tjs(self):
        """
        calculate all features of the columns
        Returns
        -------
        """

        def normalized(feature_df):
            norm_df = {}
            for feature_ele in ['uc', 'cm', 'ws', 'emc']:
                mean = sum(feature_df.values()) / len(feature_df.values())
                std_deviation = math.sqrt(sum((x - mean) ** 2 for x in feature_df.values()) / len(feature_df.values()))
                norm_df[feature_ele] = (feature_df[feature_ele] - mean) / std_deviation
            return norm_df
        self.ws_cal(top_n=3)
        for i, candidate_type in self.annotation.items():
            if candidate_type == SCD.ColumnType.named_entity:
                columnDetection = SCD.ColumnDetection(self.table.iloc[:, i], column_type=candidate_type)
                feature_dict = columnDetection.features(i, self.annotation)
                feature_dict['ws'] = self.column_score[i]
                norm = normalized(feature_dict)
                self.column_score[i] = (norm['uc'] + 2 * (norm['cm'] + norm['ws']) - norm['emc']) / (
                    math.sqrt(feature_dict['df'] + 1))
        return self.column_score
