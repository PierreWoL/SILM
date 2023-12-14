import sys
from concurrent.futures import ThreadPoolExecutor
import pickle
from argparse import Namespace
import os
import time
import numpy as np
import pandas as pd
from SubjectColDetect import subjectColumns
from Utils import mkdir, most_frequent
from TableCluster.tableClustering import column_gts
from RelationshipSearch.SimilaritySearch import group_files, entityTypeRelationship


def group_files_by_superclass(df):
    superclass_dict = {}
    for superclass in df['superclass'].unique():
        files = df[df['superclass'].apply(lambda x: x == superclass)]['fileName'].tolist()
        superclass_dict[str(superclass)] = files
    return superclass_dict


def find_labels(col_list, gtclusters):
    labels = []
    for column in col_list:
        if column in gtclusters.keys():
            label = gtclusters[column]
            if type(label) is list:
                for item_label in label:
                    labels.append(item_label)
            else:
                labels.append(label)
    if len(labels) == 0:
        cluster_label = None
    else:
        cluster_label = most_frequent(labels)
    return cluster_label


def find_conceptualAttri(cols_dict, table, attributes, gtclusters=None):
    """
    cols_dict: {'EventName': ['SOTAB_0.0', 'SOTAB_111.0', 'SOTAB_119.0',], 'organizer': []...}
    or {0: ['SOTAB_0.0', 'SOTAB_111.0', 'SOTAB_119.0',], 1: []...}
    table: table name
    attributes: attributes of table in the similarity dict

    """
    combines = [f"{table[:-4]}.{attribute}" for attribute in attributes]

    if gtclusters is None:
        conceptualAttributes = [cols_dict[i] for i in combines if i in cols_dict.keys()]
    else:
        cluster = [key for i in combines for key, value in cols_dict.items() if i in value]
        conceptualAttributes = [find_labels(cols_dict[i], gtclusters) for i in cluster]
    return conceptualAttributes


def MetricType(groundTruth, results):
    TP = [i for i in results if i in groundTruth]
    FP = [i for i in results if i not in groundTruth]
    FN = [i for i in groundTruth if i not in results]

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0
    recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0
    F1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, F1score


def relationshipDiscovery(hp: Namespace):
    timing = []
    subjectColPath = os.path.join(os.getcwd(), f"datasets/{hp.dataset}")
    data_path = os.path.join(subjectColPath, "Test")
    table_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]

    ground_truth = os.path.join(os.getcwd(), f"datasets/{hp.dataset}/groundTruth.csv")
    Ground_t = group_files(pd.read_csv(ground_truth))

    types = list(Ground_t.keys())
    print(types)
    SE = subjectColumns(subjectColPath)

    datafile_path = os.getcwd() + "/result/embedding/starmie/vectors/" + hp.dataset + "/"
    files = [fn for fn in os.listdir(datafile_path) if
             fn.endswith(
                 '.pkl') and f"_{hp.embed}_" in fn]  # and 'SCT6' in fn and 'header' not in fn
    files = [fn for fn in files if not fn.endswith("subCol.pkl")]  # and 'Pretrain' in fn and 'header' in fn

    with open(f'datasets\{hp.dataset}\Relationship_gt.pickle', 'rb') as handle:
        gt_relationship = pickle.load(handle)
    gt_attributes = []
    for key, value in gt_relationship.items():
        for atttr_p in value:
            a1 = f"{key[0]}.{atttr_p[0]}"
            a2 = f"{key[1]}.{atttr_p[1]}"
            gt_attributes.append((a1, a2))
    print(gt_relationship.keys())
    print(gt_attributes)
    print(files)



    score_path = os.path.join(os.getcwd(),
                               f"result/P4/{hp.dataset}/")
    mkdir(score_path)


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
        else:
            content = pickle.load(F)

        cluster_relationships = {}
        target_path = os.path.join(os.getcwd(),
                                   f"result/P4/{hp.dataset}/{embedding_file[:-4]}/")

        startTimeS = time.time()
        for index, type_i in enumerate(types):
            for type_j in types[index + 1:]:

                cluster1 = Ground_t[type_i]
                cluster2 = Ground_t[type_j]
                # print(type_i, type_j, cluster1, cluster2)
                #if type_i == 'Organization' and type_j == 'Person':
                cluster1_embedding = [i for i in content if i[0] in cluster1]
                cluster2_embedding = [i for i in content if i[0] in cluster2]
                relationship1 = entityTypeRelationship(cluster1_embedding, cluster2_embedding, hp.similarity,
                                                       SE)
                relationship2 = entityTypeRelationship(cluster2_embedding, cluster1_embedding, hp.similarity,
                                                       SE)
                if len(relationship1) > 0:
                    cluster_relationships[(type_i, type_j)] = relationship1
                if len(relationship2) > 0:
                    cluster_relationships[(type_j, type_i)] = relationship2

        endTimeS = time.time()
        timing.append({'Embedding File': embedding_file, "timing": endTimeS - startTimeS})

        mkdir(target_path)
        with open(os.path.join(target_path, 'Relationships.pickle'), 'wb') as handle:
            pickle.dump(cluster_relationships, handle, protocol=pickle.HIGHEST_PROTOCOL)
        timing_df = pd.DataFrame(timing)
        timing_df.to_csv(os.path.join(os.getcwd(),
                                      f"result/P4/{hp.dataset}/timing_{hp.embed}.csv"))

        result_type_pairs = list(cluster_relationships.keys())
        p, r, f1 = MetricType(list(gt_relationship.keys()), result_type_pairs)

        precisionk = attributeRelationshipSearch(cluster_relationships, hp, embedding_file, SE, gt_attributes)
        print(embedding_file, "\ntype metric\n",
              f"precision {p} recall: {r} F1-score: {f1}\n",
              f"attribute Precision@{hp.topk}: {precisionk}")

        if 'TypeRelationshipScore.csv' in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, 'TypeRelationshipScore.csv'), index_col=0)
        else:
            df = pd.DataFrame(columns=['Similarity', 'Embedding', 'Precision', 'Recall', 'F1-score'])
            df.to_csv(os.path.join(score_path, 'TypeRelationshipScore.csv'))

        if str(hp.similarity) + embedding_file[:-4] not in df.index:
            new_data = {'Similarity': hp.similarity, 'Embedding': embedding_file[:-4], "Precision": p
                , "Recall": r, "F1-score": f1}
            # print(new_data)
            new_row = pd.DataFrame([new_data], index=[str(hp.similarity) + embedding_file[:-4]])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            # print(df)
            df.to_csv(os.path.join(score_path, 'TypeRelationshipScore.csv'))
        else:
            df.loc[str(hp.similarity), 'Similarity'] = hp.similarity
            df.loc[str(hp.similarity), 'Embedding'] = embedding_file[:-4]
            df.loc[str(hp.similarity), 'Precision'] = p
            df.loc[str(hp.similarity), 'Recall'] = r
            df.loc[str(hp.similarity), 'F1-score'] = f1

            # print(df)
            df.to_csv(os.path.join(score_path, 'TypeRelationshipScore.csv'))

        if 'AttributeRelationshipScore.csv' in os.listdir(score_path):
            df = pd.read_csv(os.path.join(score_path, 'AttributeRelationshipScore.csv'), index_col=0)
        else:
            df = pd.DataFrame(columns=['Similarity', 'Embedding', 'Precision@GT'])
            df.to_csv(os.path.join(score_path, 'AttributeRelationshipScore.csv'))

        if str(hp.similarity) + embedding_file[:-4] not in df.index:
            new_data = {'Similarity': hp.similarity, 'Embedding': embedding_file[:-4], "Precision@GT": precisionk}
            # print(new_data)
            new_row = pd.DataFrame([new_data], index=[str(hp.similarity) + embedding_file[:-4]])
            # Concatenate the new DataFrame with the original DataFrame
            df = pd.concat([df, new_row])
            # print(df)
            df.to_csv(os.path.join(score_path, 'AttributeRelationshipScore.csv'))
        else:
            df.loc[str(hp.similarity), 'Similarity'] = hp.similarity
            df.loc[str(hp.similarity) , 'Embedding'] =  embedding_file[:-4]
            df.loc[str(hp.similarity) , 'Precision@GT'] =precisionk

            # print(df)
            df.to_csv(os.path.join(score_path, 'AttributeRelationshipScore.csv'))






def attributeRelationshipSearch(cluster_relationships, hp: Namespace, embedding_file, SE, gt_attributes):
    def check_conceptualAttri(GroundTruth, gt_clusterDict, gtClusters, table, subcols):
        type = [key for key, value in GroundTruth.items() if table in value][0]

        if hp.baseline is False:
            index = list(gt_clusterDict.keys()).index(type)
            checkfile = f"{index}_colcluster_dict.pickle"
            F_cluster = open(os.path.join(datafile_path, checkfile), 'rb')
            # TODO needs to change this in the table clustering -> clustering into soft coded
            col_cluster = pickle.load(F_cluster)[hp.clustering]
            gt_clusters_type = gtClusters[type]
            conceptual = find_conceptualAttri(col_cluster, table, subcols, gtclusters=gt_clusters_type)
        else:
            conceptual = find_conceptualAttri(gtClusters[type], table, subcols)
        return conceptual

    def precisionK(gt_list, result_list):
        topk = min(hp.topk, len(result_list)) if hp.topk > 0 else len(gt_list)
        resultK = [i for i in result_list[:topk] if i[0] in gt_list or (i[0][1], i[0][0]) in gt_list]  #
        # for i in resultK:
        # print(i[0])
        precision = len(resultK) / len(result_list) if len(result_list) > 0 else 0
        return precision


    # read generated conceptual attributes of the embedding methods
    gt_clusters, ground_t, gt_cluster_dict = column_gts(hp.dataset)
    datafile_path = os.path.join(os.getcwd(), "result/SILM/", hp.dataset,
                                 "All/" + embedding_file[:-4] + "/column")
    # type of the table
    ground_truth = os.path.join(os.getcwd(), f"datasets/{hp.dataset}/groundTruth.csv")
    Ground_t = group_files_by_superclass(pd.read_csv(ground_truth))

    relationships = {}

    for type_pair, table_pair_dict in cluster_relationships.items():
        relationships[type_pair] = {}
        for table_pair, similarities in table_pair_dict.items():

            t1, t2 = table_pair[0], table_pair[1]
            NE_list_1, columns1, types1 = SE[t1]
            t1_subcol = [columns1[NE_list_1[0]]] if len(NE_list_1) != 0 else [columns1[0]]
            t2_cols = list(similarities.keys())
            t1_Attribute = check_conceptualAttri(Ground_t, gt_cluster_dict, gt_clusters, t1, t1_subcol)
            t2_Attribute = check_conceptualAttri(Ground_t, gt_cluster_dict, gt_clusters, t2, t2_cols)
            # print(similarities,t2_Attribute )
            for Attri_i in t1_Attribute:
                for index, Attri_j in enumerate(t2_Attribute):
                    if (Attri_i, Attri_j) not in relationships[type_pair]:
                        relationships[type_pair][(Attri_i, Attri_j)] = [list(similarities.values())[index]]
                    else:
                        relationships[type_pair][(Attri_i, Attri_j)].append(list(similarities.values())[index])

    overall_results = {}
    for type_pair, attribute_pair_dict in relationships.items():
        for attribute_pair, attribute_similarities in attribute_pair_dict.items():
            attri_combine1 = f"{type_pair[0]}.{attribute_pair[0]}"
            attri_combine2 = f"{type_pair[1]}.{attribute_pair[1]}"
            sim_attri = sum(attribute_similarities) / len(attribute_similarities)
            overall_results[(attri_combine1, attri_combine2)] = sim_attri
    sort_results = sorted(overall_results.items(), key=lambda item: item[1], reverse=True)
    for i in sort_results:
        print(i)
    precision = precisionK(gt_attributes, sort_results)
    return precision


"""
# TODO this needs to delete later
def find_conceptualAttribute(gt_col, type, table, attributes):
    combines = [f"{table[:-4]}.{attribute}" for attribute in attributes]
    conceptualAttributes = [gt_col[type][i] for i in combines if i in gt_col[type].keys()]
    return conceptualAttributes


def a(dataset):
    ground_t = column_gts(dataset)[0]

    ground_truth = os.path.join(os.getcwd(), f"datasets/{dataset}/groundTruth.csv")
    Ground_t = group_files_by_superclass(pd.read_csv(ground_truth))

    subjectColPath = os.path.join(os.getcwd(), f"datasets/{dataset}")
    SE = subjectColumns(subjectColPath)
    target_path = os.path.join(os.getcwd(),
                               f"result/P4/{dataset}/Pretrain_bert_head_column_header_False/")  # Pretrain_sbert_head_column_header_False
    if os.path.exists(os.path.join(target_path, 'Relationships.pickle')):
        with open(os.path.join(target_path, 'Relationships.pickle'), 'rb') as handle:
            cluster_relationships = pickle.load(handle)

    column_relationships = []
    for type_pair, table_pair_dict in cluster_relationships.items():
        for table_pair, similarities in table_pair_dict.items():
            # print(table_pair, similarities)
            t1 = table_pair[0]
            t2 = table_pair[1]
            type_1_list = [key for key, value in Ground_t.items() if t1 in value][0]
            type_2_list = [key for key, value in Ground_t.items() if t2 in value][0]
            NE_list_1, columns1, types1 = SE[t1]
            t1_subcol = columns1[NE_list_1[0]] if len(NE_list_1) != 0 else columns1[0]
            t2_cols = list(similarities.keys())

            t1_conceptual = find_conceptualAttribute(ground_t, type_1_list, t1, [t1_subcol])
            if len(t1_conceptual) == 0:
                concept = t1_subcol
            else:
                concept = t1_conceptual[0]
            t2_conceptual = find_conceptualAttribute(ground_t, type_2_list, t2, t2_cols)
            for t2_col_concept in t2_conceptual:
                column_relationships.append(
                    {'Type1': type_pair[1], 'SubAttribute': concept, 'table1': t1, 'subcol': t1_subcol,
                     'T2': type_pair[0], 'Attribute': t2_col_concept, 'table2': t2})
            if len(t2_cols) > 1:
                print(
                    f"t1 {t1} t2{t2} t1 subject attributes {t1_conceptual} t2 cols {t2_conceptual} {similarities.values()}")
    column_relationships = pd.DataFrame(column_relationships)
    column_relationships.to_csv(f"datasets/{dataset}/columnRelationships.csv", index=False)
    # types.to_csv("datasets/WDC/clusterRelationships.csv",index=False)
"""
