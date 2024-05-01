import os
import pickle
import sys

import numpy as np
import pandas as pd
from Utils import calculate_similarity
sys.path.append('../')
import SCDection.TableAnnotation as TA
from SCDection.SubjectColumnDetection import ColumnType


# Assuming that embeddings are stored in a dictionary in this format:
# embeddings = { 'subject1': [...], 'A': [...], ... }



def entityTypeRelationship(cluster1, cluster2, threshold1, EntityColumns,  dataset,isEuclidean=False):
    table_pairs = {}
    for table_1, table1_embedding in cluster1:
        annotation, NE_column_score = EntityColumns[table_1]
        if len(NE_column_score) > 0:
            max_score = max(NE_column_score.values())
            subcol_index = [key for key, value in NE_column_score.items() if value == max_score]
        else:
            subcol_index = [0]
        subjectCol_1_embedding = table1_embedding[subcol_index[0]]
        for table_2, table2_embedding in cluster2:
            similar_pairs = {}
            annotation2, NE_column_score2 = EntityColumns[table_2]
            left_index = []
            if len(NE_column_score2) > 0:
                max_score2 = max(NE_column_score2.values())
                left_index = [key for key, value in NE_column_score2.items() if value != max_score2]
            NE_embeddings2 = np.array([table2_embedding[i] for i in left_index]) if len(
                left_index) != 0 else table2_embedding
            columns2 = pd.read_csv(os.path.join(os.getcwd(),f"datasets/{dataset}/Test/", table_2)).columns
            NE_columns = [columns2[i] for i in left_index] if len(left_index) != 0 else columns2


            for index, j in enumerate(NE_embeddings2):
                similarity = calculate_similarity(subjectCol_1_embedding, j, Euclidean=isEuclidean)
                should_add = (not isEuclidean and similarity > threshold1) or (isEuclidean and similarity < threshold1)
                if should_add:
                    similar_pairs[NE_columns[index]] = similarity

            if len(similar_pairs) != 0:
                table_pairs[(table_1, table_2)] = similar_pairs

    return table_pairs


def table_relationship(df1_embedding, df2_embedding, SubjectAttri: dict, similarity_dict: dict):
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




