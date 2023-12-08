import sys
from concurrent.futures import ThreadPoolExecutor
import pickle
from argparse import Namespace
import os
import time

import numpy as np
import pandas as pd
from SubjectColDetect import subjectColumns
from Utils import mkdir

from RelationshipSearch.SimilaritySearch import group_files, entityTypeRelationship


def relationshipDiscovery(hp: Namespace):
    timing = []
    subjectColPath = os.path.join(os.getcwd(), f"datasets/{hp.dataset}")
    data_path = os.path.join(subjectColPath, "Test")
    table_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]

    ground_truth = os.path.join(os.getcwd(), f"datasets/{hp.dataset}/groundTruth.csv")
    Ground_t = group_files(pd.read_csv(ground_truth))
    types = list(Ground_t.keys())
    SE = subjectColumns(subjectColPath)

    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if
             fn.endswith('.pkl') and f"_{hp.embed}_" in fn]  # and 'SCT6' in fn and 'header' not in fn
    files = [fn for fn in files if not fn.endswith("subCol.pkl")]

    print(files, len(files))
    for embedding_file in files:

        F = open(os.path.join(datafile_path, embedding_file), 'rb')
        if embedding_file.endswith("_column.pkl"):
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
            #print(content)
        else:
            content = pickle.load(F)

        startTimeS = time.time()
        cluster_relationships = {}
        target_path = os.path.join(os.getcwd(),
                                   f"result/P4/{hp.dataset}/{embedding_file[:-4]}")

        for index, type_i in enumerate(types):
                for type_j in types[index+1:]:
                    cluster2 = Ground_t[type_i]
                    cluster1 = Ground_t[type_j]

                    cluster1_embedding = [i for i in content if i[0] in cluster1]
                    cluster2_embedding = [i for i in content if i[0] in cluster2]
                    relationship1 = entityTypeRelationship(cluster1_embedding, cluster2_embedding, 0.6, SE)

                    relationship2 = entityTypeRelationship(cluster2_embedding, cluster1_embedding, 0.6, SE)
                    if len(relationship1) > 0:
                        cluster_relationships[(type_i, type_j)] = relationship1
                    if len(relationship2) > 0:
                        cluster_relationships[(type_j, type_i)] = relationship2
        print(cluster_relationships)
        endTimeS = time.time()
        timing.append({'Embedding File': embedding_file, "timing": endTimeS-startTimeS})

        mkdir(target_path)
        with open(os.path.join(target_path, 'Relationships.pickle'), 'wb') as handle:
                pickle.dump(cluster_relationships, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_df = pd.DataFrame(timing)
    timing_df.to_csv(os.path.join(os.getcwd(),
                                   f"result/P4/{hp.dataset}/timing_{hp.embed}.csv"))
