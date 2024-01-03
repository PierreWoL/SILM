import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def rand_index(predicted_labels, ground_truth_labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    all = 0
    for i in range(len(predicted_labels)):
        i_predict = predicted_labels[i]
        i_true = ground_truth_labels[i]
        for j in range(i, len(predicted_labels)):
            if i != j:
                j_predict = predicted_labels[j]
                j_true = ground_truth_labels[j]
                jaccard_sim_predict = jaccard_similarity(set(i_predict), set(j_predict))
                jaccard_sim_true = jaccard_similarity(set(i_true), set(j_true))
                if jaccard_sim_predict > 0 and jaccard_sim_true > 0:
                    true_positive += 1
                # print(all,i_predict,j_predict, "and ground truth", i_true,j_true )
                elif jaccard_sim_predict == 0 and jaccard_sim_true == 0:
                    true_negative += 1
                elif jaccard_sim_predict > 0 and jaccard_sim_true == 0:
                    # print(all, i_predict,j_predict, "and ground truth", i_true,j_true )
                    false_positive += 1
                elif jaccard_sim_predict == 0 and jaccard_sim_true > 0:
                    false_negative += 1
                all += 1
    print((true_positive + true_negative), all)
    RI = (true_positive + true_negative) / all
    return RI


from itertools import combinations


def calculate_rand_index(predicted_labels, ground_truth_labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    all=0

    for i in range(len(predicted_labels)):
        for j in range(i,len(predicted_labels)):
            if i != j:
                all+=1
                same_cluster_predicted = set(predicted_labels[i]) & set(predicted_labels[j])

                same_cluster_ground_truth = set(ground_truth_labels[i]) & set(ground_truth_labels[j])
                #print(all, predicted_labels[i], predicted_labels[j], same_cluster_predicted,"and ground truth", same_cluster_ground_truth)
                if len(same_cluster_predicted)>0 and len(same_cluster_ground_truth)>0:
                    true_positive += 1
                elif  len(same_cluster_predicted) ==0 and  len(same_cluster_ground_truth)==0:
                    true_negative += 1
                elif len(same_cluster_predicted)>0 and  len(same_cluster_ground_truth) ==0:
                    false_positive += 1
                elif  len(same_cluster_predicted)==0 and len(same_cluster_ground_truth)>0:
                    false_negative += 1
    print((true_positive + true_negative),all )
    rand_index = (true_positive + true_negative) / all
    return rand_index

cluster1 = [['A', 'B'], ['C', 'D'], ['E']]
cluster2 = [['A', 'D'], ['C'], ['B', 'E']]

predicted_labels = [['Place', 'Organization'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place', 'Organization'], ['Place'], ['Place'], ['Organization'], ['Place', 'Organization'], ['Organization'], ['Place', 'Organization'], ['Place'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['Event'], ['Person'], ['CreativeWork'], ['Place'], ['Person'], ['Place'], ['CreativeWork'], ['Place'], ['Organization'], ['Place'], ['Place', 'Organization'], ['Person'], ['Place', 'Organization'], ['Place'], ['Place', 'Organization'], ['Event'], ['Intangible'], ['Organization'], ['Organization'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Place'], ['Event'], ['Place', 'Organization'], ['Place'], ['Place', 'Organization'], ['Place'], ['Organization'], ['Person'], ['Place', 'Organization'], ['Place'], ['Place'], ['CreativeWork'], ['Event'], ['Place'], ['Organism'], ['Place'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Place', 'Organization'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Person'], ['Event'], ['Place'], ['Event'], ['Place'], ['Event'], ['Place', 'Organization'], ['Place', 'Organization'], ['CreativeWork', 'Intangible'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['Animal'], ['CreativeWork'], ['CreativeWork'], ['Place', 'Organization'], ['Intangible'], ['Place'], ['CreativeWork'], ['Place'], ['Place'], ['Person'], ['Place'], ['Place'], ['Place'], ['Place'], ['Organization'], ['Place', 'Organization'], ['Animal'], ['Intangible'], ['Event'], ['Intangible'], ['Event'], ['Organization'], ['Event'], ['Organization'], ['Place'], ['CreativeWork'], ['Animal'], ['Person'], ['Place'], ['Place', 'Organization'], ['Place'], ['Person'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['Place'], ['CreativeWork', 'Intangible'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Organization'], ['Place', 'Organization'], ['Place', 'Organization'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place'], ['Event'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Animal'], ['CreativeWork'], ['CreativeWork'], ['Animal'], ['Organization'], ['CreativeWork'], ['Place'], ['Place'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Event'], ['Place'], ['Animal'], ['Intangible'], ['Place'], ['Organization'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['Organization'], ['Place'], ['Animal'], ['Place', 'Organization'], ['CreativeWork'], ['CreativeWork', 'Intangible'], ['Organization'], ['Place'], ['Place'], ['Place', 'Organization'], ['Person'], ['Place'], ['Event'], ['Place'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place'], ['Animal'], ['Place'], ['Place', 'Organization'], ['Event'], ['Person'], ['Place'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place', 'Organization'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Place'], ['Place'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['Place'], ['Place'], ['CreativeWork', 'Intangible'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Place'], ['Intangible'], ['CreativeWork'], ['Person'], ['Place'], ['Organism'], ['CreativeWork'], ['CreativeWork'], ['Event'], ['Animal'], ['Place', 'Organization'], ['Place'], ['Place', 'Organization'], ['Place'], ['Place'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['Place', 'Organization'], ['Person'], ['Person'], ['Place'], ['Place', 'Organization'], ['Place'], ['Event'], ['Organization'], ['Event'], ['Event'], ['Place'], ['Place'], ['Person'], ['Place'], ['Intangible'], ['Event'], ['Person'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['CreativeWork'], ['Place'], ['Place'], ['Place', 'Organization'], ['Organization'], ['Person'], ['Place', 'Organization'], ['Place', 'Organization'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Place'], ['Person'], ['CreativeWork'], ['Event'], ['Intangible'], ['Place'], ['Place'], ['CreativeWork'], ['Person'], ['Place'], ['Place'], ['Event'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['Event'], ['Place', 'Organization'], ['CreativeWork'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Animal'], ['Place'], ['CreativeWork'], ['Person'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place', 'Organization'], ['Event'], ['CreativeWork'], ['Event'], ['Place', 'Organization'], ['Place'], ['Place'], ['Place'], ['CreativeWork', 'Intangible'], ['Person'], ['Place', 'Organization'], ['Animal'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['Place'], ['Place'], ['Place'], ['Place', 'Organization'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['CreativeWork'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Person'], ['CreativeWork'], ['CreativeWork'], ['Event'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Person'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['Place'], ['Event'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Event'], ['Event'], ['Place'], ['Event'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Place', 'Organization'], ['Person'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Event'], ['Place', 'Organization'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Place'], ['Place'], ['Person'], ['Person'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['Event'], ['Place'], ['CreativeWork'], ['Person'], ['Place'], ['Person'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place'], ['Person'], ['CreativeWork'], ['Person'], ['Place'], ['Place'], ['CreativeWork'], ['Place'], ['Event'], ['Place'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['CreativeWork'], ['Person'], ['Place'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Person'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Person'], ['Person'], ['Place'], ['Person'], ['Person'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['Person'], ['Place'], ['Place'], ['CreativeWork'], ['Person'], ['Place'], ['Person'], ['Place'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Person'], ['CreativeWork'], ['Person'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place', 'Organization'], ['Person'], ['Place'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Person'], ['Person'], ['Place', 'Organization'], ['Event'], ['Place', 'Organization'], ['Place'], ['Place'], ['CreativeWork'], ['Person'], ['CreativeWork'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place', 'Organization'], ['Place', 'Organization'], ['Person'], ['CreativeWork'], ['Place', 'Organization'], ['Place'], ['Place'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['CreativeWork'], ['Place', 'Organization'], ['Person'], ['Place'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Place', 'Organization'], ['Place', 'Organization'], ['CreativeWork'], ['Event'], ['Place', 'Organization'], ['Place', 'Organization'], ['Person'], ['Event'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Person'], ['Place'], ['Place', 'Organization'], ['Place', 'Organization'], ['Person'], ['Place'], ['Person'], ['Place', 'Organization'], ['Organism'], ['CreativeWork'], ['Place', 'Organization'], ['Event'], ['CreativeWork'], ['Place', 'Organization'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Person'], ['Place', 'Organization'], ['Place'], ['Person'], ['Place', 'Organization'], ['CreativeWork'], ['Person'], ['Place', 'Organization'], ['CreativeWork'], ['Place'], ['Place', 'Organization'], ['Place'], ['CreativeWork'], ['Place'], ['CreativeWork'], ['Event']]

ground_truth_labels =  [['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['CreativeWork'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place'], ['Place']]


ri_score = rand_index(predicted_labels, ground_truth_labels)
RI = calculate_rand_index(predicted_labels, ground_truth_labels)
print("Rand Index:", ri_score, RI)

embeddings = [fn for fn in os.listdir("result/embedding/TabFact/") if "roberta" in fn and "Pretrain" in fn and "True" in fn]
for embedding in embeddings:
    with open(os.path.join("result/embedding/TabFact", embedding), "rb") as file:
        G = pickle.load(file)
    new_embedding = []
    for file_tuple in G:
        list_col = []
        file_name = file_tuple[0]
        for array_col in file_tuple[1]:
            list_col.append(np.array(array_col))
        list_col = np.array(list_col)
        new_embedding.append((file_name,list_col))
    with open(os.path.join("result/embedding/TabFact", embedding), "wb") as file:
        pickle.dump(new_embedding, file)

"""
datasets = ['TabFact', "WDC"]
for dataset in datasets:
    datafile_path = os.getcwd() + "/result/embedding/" + dataset + "/"
    data_path = os.getcwd() + "/datasets/" + dataset + "/Test/"
    files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn and "roberta" in fn and "Pretrain" in fn]
    for file in files:
        dict_file = {}
        F = open(datafile_path + file, 'rb')
        content = pickle.load(F)
        update_content = []
        for i in content:
            table_name = i[0]
            vector =  np.array([np.array(i) for i in i[1] ])
            update_content.append((table_name, vector))
        with open( datafile_path + file, "wb") as file:
            pickle.dump(update_content, file)

for dataset in datasets:
    datafile_path = os.getcwd() + "/result/embedding/" + dataset + "/"
    data_path = os.getcwd() + "/datasets/" + dataset + "/Test/"
    files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn and "roberta" in fn and "Pretrain" in fn]
    for file in files:
        dict_file = {}
        F = open(datafile_path + file, 'rb')
        content = pickle.load(F)
        print(content)
"""