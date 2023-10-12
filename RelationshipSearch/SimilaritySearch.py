import os
import pickle

import numpy as np
import pandas as pd



# Assuming that embeddings are stored in a dictionary in this format:
# embeddings = { 'subject1': [...], 'A': [...], ... }
def calculate_similarity(e1, e2):
    dot_product = np.dot(e1, e2)

    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)

    cosine_similarity = dot_product / (norm_e1 * norm_e2)
    return cosine_similarity


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


datafile_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/result/embedding/starmie/vectors/WDC/"
table_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/datasets/WDC/"

file = "cl_sample_row_lm_sbert_head_column_0_none.pkl"
F = open(datafile_path + file, 'rb')
content = pickle.load(F)

import TableAnnotation as TA
from SubjectColumnDetection import ColumnType
example_embedding = {i[0]:i[1] for i in content}
subjectAttri_datasets = subjectAttri(table_path, example_embedding)
df1_example = "SOTAB_125.csv",example_embedding["SOTAB_125.csv"]
df2_example = "SOTAB_200.csv",example_embedding["SOTAB_200.csv"]

relations = table_relationship(df1_example, df2_example, subjectAttri_datasets)
print(relations)
