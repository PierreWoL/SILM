import os
import pickle
import sys

import numpy as np
import pandas as pd
sys.path.append('../')
import TableAnnotation as TA
from SubjectColumnDetection import ColumnType


# Assuming that embeddings are stored in a dictionary in this format:
# embeddings = { 'subject1': [...], 'A': [...], ... }
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
        for table_2, table2_embedding in cluster2:
            similar_pairs = {}
            NE_list_2, columns2, types2 = EntityColumns[table_2]
            NE_embeddings2 = np.array([table2_embedding[i] for i in NE_list_2[1:]]) if len(
                NE_list_2) != 0 else table2_embedding
            NE_columns = [columns2[i] for i in NE_list_2[1:]] if len(NE_list_2) != 0 else columns2
            for index, j in enumerate(NE_embeddings2):
                similarity = calculate_similarity(subjectCol_1_embedding, j)
                eu_sim = np.linalg.norm(subjectCol_1_embedding - j)
                # if similarity>0.6:
                    # print(subcol1, NE_columns[index], eu_sim, similarity)
                if similarity > threshold:
                    similar_pairs[NE_columns[index]] = similarity
            # if len(similar_pairs)==0:
            # similar_pairs[most_similar_col] = highest_sim
            if len(similar_pairs) != 0:
                table_pairs[(table_1, table_2)] = similar_pairs
    return table_pairs




def table_relationship(df1_embedding, df2_embedding, SubjectAttri: dict, similarity_dict:dict):
    """
    Search if table1 and table2 have highly similar attributes  (embeddings), one of which
    must be subject attributes of the belonging tables.


    Args:
        df1_embedding (tuple): the input table1's name and embedding
        df2_embedding (tuple): the input table2's name and embedding
        df_embedding looks like:
            'T2DV2_87.csv': array([[ 0.05610702,  0.05640658, -0.04423186, ...,  0.03084791,
         0.05204732,  0.01208988],[...]], dtype=float32)
        SubjectAttri :dictionary that stores the subject attributes index of each table


    Return:
        dictionary: {df1_name: (subjectAttri, []),df2_name:(subjectAttri, [])}
    """

    def column_alignment(index_list, index_list2, embedding1, embedding2, threshold=0.5):
        highest_sim = 0
        backup_ones = 0
        relationship_ones = {}
        for index, embed in enumerate(embedding2):
            for embed1 in embedding1[index_list]:
                if index not in index_list2:
                    similarity = calculate_similarity(embed1, embed)
                    print(similarity, index_list, index)
                    if similarity > highest_sim:
                        highest_sim = similarity
                        backup_ones = index
                    if similarity > threshold:
                        relationship_ones[similarity] = index
        if len(relationship_ones) < 1:
            relationship_ones[highest_sim] = backup_ones

        return relationship_ones

    df1_name, df2_name = df1_embedding[0], df2_embedding[0]
    df1_SubColI = SubjectAttri[df1_name]
    df2_SubColI = SubjectAttri[df2_name]

    df2_relatedPairs = column_alignment(df1_SubColI, df2_SubColI, df1_embedding[1], df2_embedding[1], 0.25)
    print("now df2")
    similarity_dict[df1_name][1].append()




def subjectAttri(dataset_path, Embedding):
    target = os.path.join(dataset_path, "SubjecAttributes.pkl")
    if os.path.exists(target):
        F = open(os.path.join(dataset_path, "SubjecAttributes.pkl"), 'rb')
        subjectAttributes = pickle.load(F)
    else:
        subjectAttributes = {}
        table_names = Embedding.keys()
        for table_id in table_names:
            table = pd.read_csv(os.path.join(dataset_path, "Test", table_id))
            anno = TA.TableColumnAnnotation(table)
            subjectCol_keys = []
            types = anno.annotation
            for key, type in types.items():
                if type == ColumnType.named_entity:
                    subjectCol_keys.append(key)
                    break
            if len(subjectCol_keys) == 0:
                subjectCol_keys = [i for i in range(len(table.columns))]
            subjectAttributes[table_id] = subjectCol_keys
            del table, anno, types, subjectCol_keys
        with open(target, "wb") as f:
            pickle.dump(subjectAttributes, f)
    return subjectAttributes


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

def check_similarity(cluster_1, cluster_2, embeddings: dict):
    """
    Search if table1 and table2 have highly similar attributes  (embeddings), one of which
    must be subject attributes of the belonging tables.


    Args:
        cluster_1 (list): the input table1's name and embedding
        cluster_2 (list): the input table2's name and embedding
        embeddings :dictionary that stores the subject attributes index of each table


    Return:
        dictionary: {df1_name: (subjectAttri, []),df2_name:(subjectAttri, [])}
    """

"""
datafile_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/result/embedding/starmie/vectors/WDC/"
table_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/datasets/WDC/"
similarity_dict = {}
file = "Pretrain_sbert_head_column_header_False.pkl"
F = open(datafile_path + file, 'rb')
content = pickle.load(F)

example_embedding = {i[0]:i[1] for i in content}
subjectAttri_datasets = subjectAttri(table_path, example_embedding)
df1_example = "SOTAB_125.csv",example_embedding["SOTAB_125.csv"]
df2_example = "SOTAB_200.csv",example_embedding["SOTAB_200.csv"]

relations = table_relationship(df1_example, df2_example,subjectAttri_datasets,similarity_dict)
print(relations)
"""