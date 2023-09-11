import os
import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from TableCluster.tableClustering import column_gts
from clustering import clustering_results, data_classes, clusteringColumnResults


# Assuming that embeddings are stored in a dictionary in this format:
# embeddings = { 'subject1': [...], 'A': [...], ... }

def check_similarity(cluster_1, cluster_2, embeddings):
    results = {}

    # Iterate over each dataframe in cluster_1
    for df in cluster_1:
        subject_column = df.columns[0]  # Assuming subject column is the first column
        subject_embedding = embeddings[subject_column]

        # Compare with each column of each dataframe in cluster_2
        for other_df in cluster_2:
            for col in other_df.columns:
                col_embedding = embeddings[col]
                similarity = cosine_similarity([subject_embedding], [col_embedding])[0][0]

                if similarity > 0.5:
                    results[(subject_column, col)] = "Similar"
                else:
                    results[(subject_column, col)] = "Not Similar"

    return results

# Sample dataframes
df1 = pd.DataFrame({'subject1': [1, 2], 'A': [3, 4], 'B': [5, 6]})
df2 = pd.DataFrame({'subject2': [7, 8], 'C': [9, 10], 'D': [11, 12]})
df3 = pd.DataFrame({'subject3': [13, 14], 'E': [15, 16], 'F': [17, 18]})

cluster_1 = [df1, df2, df3]
df4 = pd.DataFrame({'subject4': [19, 20], 'G': [1, 2.2]})
df5 = pd.DataFrame({'subject5': [23, 24], 'H': [7, 7.8]})

cluster_2 = [df4, df5]
embeddings = {
    # Sample embeddings, replace these with actual embeddings
    'subject1': [1, 0],
    'A': [0.9, 0.1],
    'B': [0.8, 0.2],
    'subject2': [0.7, 0.3],
    'C': [0.6, 0.4],
    'D': [0.5, 0.5],
    'subject3': [0.4, 0.6],
    'E': [0.3, 0.7],
    'F': [0.2, 0.8],
    'subject4': [0.1, 0.9],
    'G': [0, 1],
    'subject5': [1, 1],
    'H': [0.5, 0.5]
}

print(check_similarity(cluster_1, cluster_2, embeddings))