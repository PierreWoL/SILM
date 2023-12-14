import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append('../')



def calculate_similarity(e1, e2):
    dot_product = np.dot(e1, e2)

    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)

    cosine_similarity = dot_product / (norm_e1 * norm_e2)
    return cosine_similarity


def entityTypeRelationship(cluster1, cluster2, threshold, EntityColumns):
    table_pairs = {}
    for table_1, table1_embedding in cluster1:
        NE_list_1, columns1, types1 = EntityColumns[table_1]
        subjectCol_1_embedding = table1_embedding[NE_list_1[0]] if len(NE_list_1) != 0 else table1_embedding[0]
        subcol1= columns1[NE_list_1[0]] if len(NE_list_1) != 0 else columns1[0]
        for table_2, table2_embedding in cluster2:
            similar_pairs = {}

            NE_list_2, columns2, types2 = EntityColumns[table_2]

            NE_embeddings2 = np.array([table2_embedding[i] for i in NE_list_2]) if len(
                NE_list_2) != 0 else table2_embedding
            NE_columns = [columns2[i] for i in NE_list_2] if len(NE_list_2) != 0 else columns2
            for index, j in enumerate(NE_embeddings2):
                similarity = calculate_similarity(subjectCol_1_embedding, j)

                if similarity > threshold:
                    similar_pairs[NE_columns[index]] = similarity


            # if len(similar_pairs)==0:
            # similar_pairs[most_similar_col] = highest_sim
            if len(similar_pairs) != 0:
                table_pairs[(table_1, table_2)] = similar_pairs
    return table_pairs




def group_files(df):
    df['superclass'] = df['superclass'].apply(eval)
    # Create an empty dictionary to store the grouped file names
    grouped_files = {}
    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # Handle rows with multiple superclasses
        if len(row['superclass']) > 1:
            # If the class is one of the superclasses, choose it, otherwise choose the first superclass
            chosen_superclass = row['class'] if row['class'] in row['superclass'] else row['superclass'][0]
            grouped_files.setdefault(chosen_superclass, []).append(row['fileName'])
        else:
            # For rows with a single superclass, simply add the fileName to the corresponding list
            superclass_str = row['superclass'][0]  # Since we have lists of single items, we take the first one
            grouped_files.setdefault(superclass_str, []).append(row['fileName'])
    return grouped_files


"""

dataset = 'WDC'
datafile_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             f"result/embedding/starmie/vectors/{dataset}/")
embedding_file = "Pretrain_sbert_head_column_header_False.pkl"
F = open(os.path.join(datafile_path, embedding_file), 'rb')
content = pickle.load(F)

subjectColPath = os.path.join(os.path.dirname(os.getcwd()), f"datasets/{dataset}")
 
ground_truth = os.path.join(os.path.dirname(os.getcwd()), f"datasets/{dataset}/groundTruth.csv")

Ground_t = group_files(pd.read_csv(ground_truth))
print(Ground_t)
types = list(Ground_t.keys())
SE = subjectColumns(subjectColPath)
cluster_relationships = {}
for index, type_i in enumerate(types):
    for type_j in types[index:]:
        cluster2 = Ground_t[type_i]
        cluster1 = Ground_t[type_j]

        cluster1_embedding = [i for i in content if i[0] in cluster1]
        cluster2_embedding = [i for i in content if i[0] in cluster2]
        relationship1 = entityTypeRelationship(cluster1_embedding, cluster2_embedding, 0.6, SE)

        relationship2 = entityTypeRelationship(cluster2_embedding, cluster1_embedding, 0.6, SE)
        if len(relationship1)>0:
            cluster_relationships[(type_i, type_j)] = relationship1
        if len(relationship2)>0:
            cluster_relationships[(type_j, type_i)] = relationship2
#print(cluster_relationships)
target_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           f"result/P4/{dataset}/{embedding_file[:-4]}")
mkdir(target_path)
with open(os.path.join(target_path, 'Relationships.pickle'), 'wb') as handle:
    pickle.dump(cluster_relationships, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""