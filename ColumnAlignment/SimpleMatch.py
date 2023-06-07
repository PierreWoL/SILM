import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from Dataset_dict import col_concate
from operator import itemgetter

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}

"""def jaccard_similarity(vector1, vector2):
    intersection = np.logical_and(vector1, vector2)
    union = np.logical_or(vector1, vector2)
    jaccard = np.sum(intersection) / np.sum(union)
    return jaccard
"""
import torch
from sklearn.metrics.pairwise import cosine_similarity


def dataframe_slice(table: pd.DataFrame):
    table_slice = table
    if len(table) > 1000:
        table_slice = table.sample(n=1000, random_state=42)
    return table_slice


def cos_similarity(vectors):
    normalized_embeddings = torch.nn.functional.normalize(vectors, p=2, dim=1)
    similarity = cosine_similarity(normalized_embeddings[0].detach().numpy().reshape(1, -1),
                                   normalized_embeddings[1].detach().numpy().reshape(1, -1)).item()

    return similarity


class SimpleColumnMatch:
    def __init__(self, eval_path, method):
        lm = 'roberta'
        self.eval_path = eval_path
        if method == "M1" or method == "M1H":
            lm = 'roberta'
        if method == "M2" or method == "M2H":
            lm = 'sbert'
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.model = AutoModel.from_pretrained(lm_mp[lm])

    def encodings(self, text1, text2, ):
        encoded_input = self.tokenizer([text1, text2], max_length=512, padding=True, truncation=True,
                                       return_tensors='pt')
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state
        return embeddings

    def encoding(self, text1):
        encoded_input = self.tokenizer(text1, max_length=512, padding=True, truncation=True, return_tensors='pt')
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state
        return embeddings

    def embeddings_columns(self, table: pd.DataFrame,type='column'):
        cols = []
        if type =='column':
            for column_i in table.columns:
                col_i = col_concate(table[column_i], token=False)
                encoding_i = self.encoding(col_i)
                cols.append(encoding_i)
        else:
            for column_i in table.columns:
                encoding_i = self.encoding(column_i)
                cols.append(encoding_i)
        return cols

    def SimpleMatch(self, thre,type='column'):
        scores = {}
        tables = [fn for fn in os.listdir(self.eval_path) if '.csv' in fn]
        if len(tables) != 2:
            print("Wrong schema pair folder! Please check")
            return scores

        else:
            table1 = pd.read_csv(os.path.join(self.eval_path, tables[0]))
            table2 = pd.read_csv(os.path.join(self.eval_path, tables[1]))
            if type =='column':
                table1 = dataframe_slice(table1)
                table2 = dataframe_slice(table2)
            cols_1 = self.embeddings_columns(table1,type)
            cols_2 = self.embeddings_columns(table2, type)
            for column_i in cols_1:
                score_i = []
                for column_j in cols_2:
                    score = cos_similarity(self.encodings(column_i, column_j))
                    score_i.append(score)
                for score_col in score_i:
                    if score_col >= thre:
                        index_j = score_i.index(score_col)
                        scores[
                            (('table_1', column_i.rstrip()), ('table_2', table2.columns[index_j].rstrip()))] = score_col
                    else:
                        continue
            scores = dict(sorted(scores.items(), key=itemgetter(1), reverse=True))
            return scores




