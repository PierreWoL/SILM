data_path='Test01/'
gt_clusters='Test01/Test01_clusters.txt' # Ground Truth Clusters
from d3l.input_output.dataloaders import CSVDataLoader
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
import os


def create_or_find_indexes(data_path):
    #  collection of tables
    dataloader = CSVDataLoader(
        root_path=(data_path),
        # sep=",",
        encoding='latin-1'
    )

    '''
     # NameIndex
    


    '''
    if os.path.isfile(os.path.join(data_path, './name.lsh')):
        name_index = unpickle_python_object(os.path.join(data_path, './name.lsh'))
    else:
        name_index = NameIndex(dataloader=dataloader)
        pickle_python_object(name_index, os.path.join(data_path, './name.lsh'))

    # FormatIndex
    if os.path.isfile(os.path.join(data_path, './format.lsh')):
        format_index = unpickle_python_object(os.path.join(data_path, './format.lsh'))
    else:
        format_index = FormatIndex(dataloader=dataloader)
        pickle_python_object(format_index, os.path.join(data_path, './format.lsh'))

        # ValueIndex
    if os.path.isfile(os.path.join(data_path, './value.lsh')):
        value_index = unpickle_python_object(os.path.join(data_path, './value.lsh'))
    else:
        value_index = ValueIndex(dataloader=dataloader)
        pickle_python_object(value_index, os.path.join(data_path, './value.lsh'))

    # DistributionIndex
    if os.path.isfile(os.path.join(data_path, './distribution.lsh')):
        distribution_index = unpickle_python_object(os.path.join(data_path, './distribution.lsh'))
    else:
        distribution_index = DistributionIndex(dataloader=dataloader)
        pickle_python_object(distribution_index, os.path.join(data_path, './distribution.lsh'))

    return [name_index,format_index, value_index, distribution_index]  #



import os
def get_files(data_path):
    T = os.listdir(data_path)
    T = [t[:-4] for t in T if t.endswith('.csv')]
    T.sort()
    return(T)


import statistics
def initialise_distance_matrix(dim, L, dataloader, data_path, indexes):

    D = np.ones ((dim, dim))
    T = get_files(data_path)

    #Things are the same as themselves
    for i in range(dim):
        D[i,i] = 0

    for t in T:
        # qe = QueryEngine(name_index, format_index, value_index, distribution_index)
        qe = QueryEngine(*indexes)

        Neighbours = qe.table_query(table=dataloader.read_table(table_name=t),
            aggregator=None, k=10)

        for n in Neighbours: #index
            (name, similarities) = n   #'car', [1.0, 0.4, 0.4, 0.0]

            if (name in L and t != name):
                D[L[t], L[name]] = 1 - statistics.mean(similarities)
                D[L[name], L[t]] = 1 - statistics.mean(similarities)

    return D


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from time import time


def dbscan_param_search(data_path, indexes):
    dataloader = CSVDataLoader(root_path=(data_path), encoding='latin-1')

    T = get_files(data_path)

    L = {}
    for i, t in enumerate(T):
        L[t] = i

    # before_distance_matrix = time()

    D = initialise_distance_matrix(len(T), L, dataloader, data_path, indexes)

    # after_distance_matrix = time()
    # print("Building distance matrix took ",{after_distance_matrix-before_distance_matrix}," sec to run.")

    Z = pd.DataFrame(D)

    # Defining the list of hyperparameters to try
    eps_list = np.arange(start=0.1, stop=5, step=0.1)
    min_sample_list = np.arange(start=2, stop=25, step=1)

    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data = pd.DataFrame()
    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:
            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(Z)

            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                sil_score = metrics.silhouette_score(Z, labels)
            else:
                continue
            trial_parameters = "eps:" + str(eps_trial.round(1)) + " min_sample :" + str(min_sample_trial)

            silhouette_scores_data = silhouette_scores_data.append(
                pd.DataFrame(data=[[sil_score, eps_trial.round(1), min_sample_trial]],
                             columns=["score", "eps", "min_sample"]))

    # Finding out the best hyperparameters with highest Score
    par = silhouette_scores_data.sort_values(by=['score'], axis=0, ascending=False).head(1)
    eps = par.iloc[0][1]
    min_sample = par.iloc[0][2]
    return eps, int(min_sample), Z, T

def cluster_discovery(data_path,parameters):

    db = DBSCAN(eps=parameters[0],
               min_samples=parameters[1],
               n_jobs=-1)

    db.fit(parameters[2])
    labels = list(db.labels_)
    l = (parameters[3],labels)
    clust = zip(*l)
    clu = list(clust)

    return clu


def compute_rand_score(clusters_list,file):

    file = open(file,'r')
    content = file.read().strip()
    content_list = content.split(",")
    file.close()

    label=[]
    for member in clusters_list:
        label.append(member[1])

    rand_score = metrics.rand_score(content_list, label)

    return rand_score
'''
from itertools import chain, combinations
import pandas as pd
import os
import glob


def key_options(items):
        return chain.from_iterable(combinations(items, r) for r in range(1, 2))
        # return chain.from_iterable(combinations(items, r) for r in range(1, len(items)+1))

def primary_key_discovery(data_path):
    PK=[]
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    for f in csv_files:
        dataframes = pd.read_csv(f, dtype='unicode', encoding='latin-1')


        # print('Filename ',f)
        
        for candidate in key_options(list(dataframes)[0:]): #columns names
            deduped = dataframes.drop_duplicates(candidate)
            # print('    Candidate:', candidate,': len(dataframes.index) = ', len(dataframes.index), '; len(deduped.index) = ', len(deduped.index))
            if len(deduped.index) == len(dataframes.index):
                # x=[','.join(candidate)]
                x=list(candidate)
                PK_DF=dataframes[list(candidate)]
                PK.append((f,PK_DF))      
                
    return PK     



'''

indexes = create_or_find_indexes(data_path)
parameters = dbscan_param_search(data_path, indexes)
print(parameters)
clusters = cluster_discovery(data_path,parameters)
print(clusters)
compute_rand_score(clusters,gt_clusters)