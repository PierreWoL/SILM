import pickle
from argparse import Namespace
import os
import time
import pandas as pd
from matplotlib import pyplot as plt
from SubjectColDetect import subjectColumns
from Utils import mkdir, most_frequent
from TableCluster.tableClustering import column_gts
from RelationshipSearch.SimilaritySearch import group_files, entityTypeRelationship
import numpy as np

def relationshipDiscovery(dataset, embed):
    subjectColPath = os.path.join(os.getcwd(), f"datasets/{dataset}")
    data_path = os.path.join(subjectColPath, "Test")
    SE = subjectColumns(subjectColPath)
    table_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]
    ground_truth = os.path.join(os.getcwd(), f"datasets/{dataset}/groundTruth.csv")
    Ground_t = group_files(pd.read_csv(ground_truth))
    types = list(Ground_t.keys())
    print("types",types)
    datafile_path = os.getcwd() + "/result/embedding/" + dataset + "/"
    embedding_file = embed
    F = open(os.path.join(datafile_path, embedding_file), 'rb')
    """ if embedding_file.endswith("_column.pkl"):
        original_content = pickle.load(F)
        content = []
        for fileName in table_names:
            embedding_fileName = []
            table = pd.read_csv(os.path.join(data_path, fileName))
            for col in table.columns:
                embedding_col = [i[1] for i in original_content if i[0] == f"{fileName[:-4]}.{col}"][0]
                embedding_fileName.append(embedding_col[0])
            tuple_file = fileName, np.array(embedding_fileName)
            content.append(tuple_file)
    else:
        content = pickle.load(F)
    for index, type_i in enumerate(types):
        for type_j in types[index + 1:]:
            cluster1 = Ground_t[type_i]
            cluster2 = Ground_t[type_j]
            # if type_i == 'Organization' and type_j == 'Person':
            cluster1_embedding = [i for i in content if i[0] in cluster1]
            cluster2_embedding = [i for i in content if i[0] in cluster2]
            for table_1, table1_embedding in cluster1_embedding:
                NE_list_1, columns1, types1 = SE[table_1]
                subjectCol_1_embedding = table1_embedding[NE_list_1[0]] if len(NE_list_1) != 0 else \
                    table1_embedding[0]
"""
relationshipDiscovery("WDC", )