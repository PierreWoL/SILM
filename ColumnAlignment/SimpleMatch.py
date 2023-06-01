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


def dataframe_slice(table:pd.DataFrame):
    table_slice = table
    # 随机选择10000行作为新的DataFrame
    if len(table)>1000:
        table_slice = table.sample(n=1000, random_state=42)
    return table_slice

def cos_similarity(vectors):
    #cos = torch.nn.CosineSimilarity(dim=0)
    #similarity = cos(vectors[0], vectors[1])
    # 归一化向量
    normalized_embeddings = torch.nn.functional.normalize(vectors, p=2, dim=1)
    # 计算余弦相似度
    similarity = cosine_similarity(normalized_embeddings[0].detach().numpy().reshape(1, -1),
                                   normalized_embeddings[1].detach().numpy().reshape(1, -1)).item()

    return similarity


class SimpleColumnMatch:
    def __init__(self, eval_path, method):
        lm = 'roberta'
        self.eval_path = eval_path
        if method == "M1":
            lm = 'roberta'
        if method == "M2":
            lm = 'sbert'
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.model = AutoModel.from_pretrained(lm_mp[lm])

    def encoding(self, text1,text2,):
        encoded_input = self.tokenizer([text1, text2],max_length=512, padding=True, truncation=True, return_tensors='pt')
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
            table1 = dataframe_slice(table1)
            table2 = dataframe_slice(table2)
            print(table1,table2)
            for column_i in table1.columns:
                score_i = []
                col_i = col_concate(table1[column_i], token=False)
                for column_j in table2.columns:
                    col_j = col_concate(table2[column_j], token=False)
                    score = cos_similarity(self.encoding(col_i,col_j))
                    score_i.append(score)
                max_score = max(score_i)
                print(score_i)
                if max_score >= thre:
                    index_j = score_i.index(max_score)
                    scores[(('table_1', column_i), ('table_2', table2.columns[index_j]))] = max_score
                else:
                    continue
            scores = dict(sorted(scores.items(), key=itemgetter(1), reverse=True))
            return scores
