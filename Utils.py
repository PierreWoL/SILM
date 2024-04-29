import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from d3l.utils.functions import token_stop_word
import TableAnnotation as TA
from SubjectColumnDetection import ColumnType

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")


def are_all_numbers(values):
    cleaned_values = [val for val in values if
                      pd.notna(val) and str(val).strip() != '' and val not in ["n / a", "n/a", "N/A"]]
    return all(val.isdigit() for val in cleaned_values)


def simplify_string(augment_op):
    string_split_list = augment_op.split(",")
    simplified_elements = [''.join([word[0].upper() for word in element.split("_")]) for element in string_split_list]
    if len(set(simplified_elements)) == 1:
        return f"{simplified_elements[0]}{len(simplified_elements)}"
    else:
        return ",".join(simplified_elements)


def most_frequent(list1, isFirst=True):
    """
    count the most frequent occurring annotated label in the cluster
    """

    count = Counter(list1)
    if isFirst is True:
        return count.most_common(1)[0][0]
    else:
        most_common_elements = count.most_common()
        max_frequency = most_common_elements[0][1]
        most_common_elements_list = [element for element, frequency in most_common_elements if
                                     frequency == max_frequency]
        return most_common_elements_list


def aug(table: pd.DataFrame):
    exists = []
    for index in range(0, table.shape[1]):
        if are_all_numbers(table.iloc[:, index][0].split(",")) is False:
            exists.append(index)
    return table.iloc[:, exists]


def split(column: pd.Series):
    if "," in column:
        return column.split(",")
    elif "|" in column:
        return column.split("|")
    else:
        return column.split(" ")
        # return column.tolist()


def subjectCol(table: pd.DataFrame, combine=False):
    sub_cols_header = []
    anno = TA.TableColumnAnnotation(table, isCombine=combine)
    types = anno.annotation
    for key, type in types.items():
        if type == ColumnType.named_entity:
            sub_cols_header = [table.columns[key]]
            break
    return sub_cols_header


import nltk
from nltk.corpus import wordnet
from collections import defaultdict, Counter

def calculate_similarity(e1, e2, Euclidean=False):
    if Euclidean is False:
        dot_product = np.dot(e1, e2)
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        similarity = dot_product / (norm_e1 * norm_e2)
    else:
        similarity = np.linalg.norm(e1- e2)
    return similarity
def findSubCol(SE, table_name):
    NE_list, headers, types = SE[table_name]
    if NE_list:
        subjectName = headers[NE_list[0]]
    else:
        subjectName = None
    return subjectName


# Make sure you've downloaded the necessary resources
# nltk.download('wordnet')
# nltk.download('punkt')


# Tokenize and Simple Named Entity Recognition

# Tokenize and Named Entity Recognition (simple version)
def namedEntityRecognition(tokens):
    # Here, we consider every token as an entity for simplicity
    return set(tokens)


# Get synonyms (using WordNet)
def synonyms(entity):
    syns = set()
    for syn in wordnet.synsets(entity):
        for lemma in syn.lemmas():
            syns.add(lemma.name())
    return syns
def name_type(cluster, name_dict=None):
    if name_dict is None:
        name_i = naming(cluster, threshold=5)
    else:
        names = [str(name_dict[i + ".csv"]) for i in cluster]
        name_i = naming(names, threshold=5)
    return name_i

def naming(InitialNames, threshold=0):
    # Given data
    ### TODO top K name
    GivenEntities = {}
    for name in InitialNames:
        #tokens = nltk.word_tokenize(name)
        #Entities = namedEntityRecognition(tokens)
        Entities = token_stop_word(name)
        if Entities == ['']:
            continue
        Synonyms = set()
        for entity in Entities:
            Synonyms.update(synonyms(entity))
        GivenEntities[name] = (Entities, Synonyms)

    # Find the most frequently occurring values in Entities and Synonyms
    frequency = defaultdict(int)
    for name, (Entities, Synonyms) in GivenEntities.items():
        for entity in Entities:
            frequency[entity] += 1
        for syn in Synonyms:
            frequency[syn] += 1

    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    # Select terms that appear in more than 50% of the values
    if threshold == 0:
        threshold = len(InitialNames) / 2
    most_frequent_terms = [term for term, freq in sorted_frequency if freq > threshold]
    if len(most_frequent_terms) == 0:
        most_frequent_terms = [term for term, freq in sorted_frequency if freq > 0][:5]
    # Given data
    return most_frequent_terms
