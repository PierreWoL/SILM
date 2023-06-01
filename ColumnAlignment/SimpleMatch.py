import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from argparse import Namespace
from scipy.spatial.distance import cosine
from Dataset_dict import col_concate

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


def cosine_similarity(vector1, vector2):
    similarity = 1 - cosine(vector1, vector2)
    return similarity


class SimpleColumnMatch:
    def __init__(self, eval_path, method, lm='roberta'):
        self.eval_path = eval_path
        if method == "M1":
            lm = 'roberta'
        if method == "M2":
            lm = 'sbert'
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.model = AutoModel.from_pretrained(lm_mp[lm])

    def encoding(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state
        return embeddings

    def SimpleMatch(self, thre):
        scores = {}
        tables = [fn for fn in os.listdir(self.eval_path) if '.csv' in fn]
        if len(tables) != 2:
            print("Wrong schema pair folder! Please check")
            return scores

        else:
            table1 = pd.read_csv(os.path.join(self.eval_path, tables[0]), lineterminator='\n')
            table2 = pd.read_csv(os.path.join(self.eval_path, tables[1]), lineterminator='\n')
            for column_i in table1.columns:
                score_i = []
                col_i = col_concate(table1[column_i], token=True)
                for column_j in table2.columns:
                    col_j = col_concate(table2[column_j], token=True)
                    score = cosine_similarity(self.encoding(col_i), self.encoding(col_j))
                    score_i.append(score)
                max_score = max(score_i)
                if max_score > thre:
                    index_j = score_i.index(max_score)
                    scores[(('table_1', column_i), ('table_2', table2.columns[index_j]))] = max_score
                else:
                    continue
