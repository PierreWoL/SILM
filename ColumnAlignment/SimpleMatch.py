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


def cos_similarity(vector1,vector2):
    normalized_embeddings = [torch.nn.functional.normalize(vector1, p=2, dim=1)]
    normalized_embeddings.append(torch.nn.functional.normalize(vector2, p=2, dim=1))
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

    def encodings(self,texts:[]):
        encoded_input = self.tokenizer(texts, max_length=512, padding=True, truncation=True,
                                       return_tensors='pt')
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state
        return embeddings

    def encoding(self, text1):
        encoded_input = self.tokenizer(text1, max_length=512, padding=True, truncation=True, return_tensors='pt')
        output = self.model(**encoded_input)
        embedding = output.last_hidden_state
        return embedding

    def embeddings_columns(self, table1: pd.DataFrame, table2:pd.DataFrame, type='column'):
        cols = []
        if type =='column':
            for column_i in table1.columns:
                col_i = col_concate(table1[column_i], token=False)
                cols.append(col_i)
            for column_j in table2.columns:
                col_j = col_concate(table2[column_j], token=False)
                cols.append(col_j)
        else:
           cols=table1.columns.tolist()
           cols.append(header for header in table2.columns)
        cols_embeddings = self.encodings(cols)
        return cols_embeddings

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
            embeddings = self.embeddings_columns(table1,table2,type)
            for index_i in range(0,len(table1.columns)):
                score_i = []
                for index_j in range(0,len(table2.columns)):
                    score = cos_similarity(embeddings[index_i],embeddings[len(table1.columns)+index_j] )
                    score_i.append(score)
                for score_col in score_i:
                    if score_col >= thre:
                        index_score = score_i.index(score_col)
                        scores[
                            (('table_1', table1.columns[index_i].rstrip()),
                             ('table_2', table2.columns[index_score].rstrip()))] = score_col
                    else:
                        continue
            scores = dict(sorted(scores.items(), key=itemgetter(1), reverse=True))
            return scores




