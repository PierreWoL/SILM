
from SCDection.TableAnnotation import TableColumnAnnotation as TA
import numpy as np

def subject_scores_to_weights(scores, fallback="argmax"):
    """
    scores: list or np.ndarray, raw subject column scores for columns in one table
    fallback:
        - "argmax": if all scores <= 0, give weight 1 to the column with highest raw score
        - "uniform": if all scores <= 0, use uniform weights
    """
    scores = np.asarray(scores, dtype=float)

    # Convert negative scores to zero
    pos_scores = np.maximum(scores, 0.0)

    total = pos_scores.sum()

    if total > 0:
        return pos_scores / total

    # fallback when all scores are non-positive
    if fallback == "argmax":
        weights = np.zeros_like(scores)
        weights[np.argmax(scores)] = 1.0
        return weights

    elif fallback == "uniform":
        return np.ones_like(scores) / len(scores)

    else:
        raise ValueError("fallback must be 'argmax' or 'uniform'")


def weights_data(data_item):
    annotation_table = TA(data_item)
    annotation_table.subcol_Tjs(NE=False)  # False
    NE_column_score = annotation_table.column_score
    return subject_scores_to_weights(list(NE_column_score.values()), fallback="argmax")





"""
import pickle
import os
import pandas as pd
dataset = "WDC" #noiseLevel/80_pct attribute_overlap/D_75
name = "Pretrain_sbert_head_column_none_False.pkl" #Pretrain_gpt3_head_column_none_False.pkl  Pretrain_sbert_head_column_none_False
new_name = "Pretrain_sbert_head_column_none_weighted_False.pkl"
pkl_path = os.path.join(f"E:/Project/CurrentDataset/result/embedding/{dataset}/",name)
data_path = f"E:/Project/CurrentDataset/datasets/{dataset}/Test"
target_path =os.path.join(f"E:/Project/CurrentDataset/result/embedding/{dataset}/",new_name)

with open(pkl_path, "rb") as f:
    embeddings = pickle.load(f)

new_weighted= []
for item in embeddings:
    dataName = item[0]
    print(dataName)
    data = pd.read_csv(os.path.join(data_path, dataName),encoding='latin1')
    weights = weights_data(data)
    X = np.array(item[1])
    if not isinstance(item[1][0],np.ndarray):
        X = ([item[1][i].data[0].embedding for i in range(len(item[1]))])
    #print(X)
    weighted_X = X * weights.reshape(-1, 1)
    table_embedding = weighted_X.sum(axis=0, keepdims=True)
    new_weighted.append((item[0],table_embedding))
pickle.dump(new_weighted, open(target_path, "wb"))"""


"""
new_weighted=[]
print(embeddings)
for item in embeddings:
    dataName = item[0]
    X = ([item[1][i].data[0].embedding for i in range(len(item[1]))])
    new_weighted.append((item[0], X))
print(new_weighted)
pickle.dump(new_weighted, open(target_path, "wb"))
"""


