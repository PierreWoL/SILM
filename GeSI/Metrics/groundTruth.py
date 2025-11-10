import ast
import os
from collections import Counter
from string import digits
from typing import Optional
from sklearn import metrics

import pandas as pd


def get_files(data_path):
    if data_path.endswith('.csv'):
        features = pd.read_csv(data_path)
        T = features.iloc[:, 0]
    else:
        T = os.listdir(data_path)
        #T = [t[:-4] for t in T if t.endswith('.csv')]
        T = [t for t in T if t.endswith('.csv')]
        T.sort()
    return T


def create_gt_file(Concepts, output_path):
    T = get_files(output_path)
    gt_string = ""
    pos = 0
    for t in T:
        name = t #.strip('.csv)')
        name = name.rstrip(digits)
        cluster = Concepts.index(name)
        if pos > 0:
            gt_string = gt_string + ","
        gt_string = gt_string + str(cluster)
        pos = pos + 1

    text_file = open(output_path + "cluster_gt.txt", "w")
    n = text_file.write(gt_string)
    text_file.close()


def data_classes(data_path, groundTruth_file, superclass=True, Nochange=False):
    """
    return three dict
    Parameters
    ----------
    data_path: the path where data stores
    groundTruth_file: table class csv files
    Returns
    -------
    gt_clusters: tablename:corresponding label dictionary. e.g.{'a': 'Book', 'b': 'Newspaper',...}
    ground_t: label and its tables. e.g. {'Book': ['a'], 'Newspaper': ['b', 'c'],...}
    gt_cluster_dict: dictionary of index: label
    like {'Political Party': 0, 'swimmer': 1, ...}
    """
    gt_file = open(groundTruth_file, errors='ignore')
    ground_truth_df = pd.read_csv(gt_file)
    test_table = []
    ground_truth_df = ground_truth_df.dropna()
    dict_gt = {}
    dict_gt0 = {}
    if superclass:
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0], ground_truth_df.iloc[:, 2]))
    else:
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0], ground_truth_df.iloc[:, 1]))# str.removesuffix(".csv")

    # dict_gt = {key: ast.literal_eval(value) for key, value in dict_gt.items() if value != " " and "[" in value}
    dict_gt0 = {}
    for key, value in dict_gt.items():
        if value != " ":
            if "[" in value:
                dict_gt0[key] = ast.literal_eval(value)
            else:
                dict_gt0[key] = value
    dict_gt = dict_gt0
    test_table2 = {}.fromkeys(test_table).keys()

    def get_concept_files(files, GroundTruth, Nochange=False):
        """
        obtain the files' classes by
        mapping its name to the ground truth
        Parameters
        ----------
        files: the test data xxxx.csv
        we have the ground truth csv that has two columns
        "filename: xxxx label: ABC"
        GroundTruth: csv file stores the ground truth
        Returns
        -------

        """
        test_gt_dic = {}
        test_gt = {}
        test_duplicate = []
        i = 0
        for file in files:
            name_without_extension = file
            if GroundTruth.get(name_without_extension) is not None:
                ground_truth = GroundTruth.get(name_without_extension)
                test_gt_dic[name_without_extension] = ground_truth
                test_duplicate.append(len(test_gt_dic.keys()))
                if type(ground_truth) is list:
                    if Nochange is False:
                        for item in ground_truth:
                            if test_gt.get(item) is None:
                                test_gt[item] = []
                            test_gt[item].append(file)
                    else:
                        if test_gt.get(str(ground_truth)) is None:
                            test_gt[str(ground_truth)] = []
                        test_gt[str(ground_truth)].append(file)
                else:
                    if test_gt.get(ground_truth) is None:
                        test_gt[ground_truth] = []
                    test_gt[ground_truth].append(file)
            i += 1
        return test_gt_dic, test_gt

    gt_clusters, ground_t = get_concept_files(get_files(data_path), dict_gt, Nochange=Nochange)
    if type(list(gt_clusters.values())[0]) is list:
        if Nochange is False:
            gt_cluster_dict = {}
            for list_type in gt_clusters.values():
                for item in list_type:
                    if item not in gt_cluster_dict.keys():
                        gt_cluster_dict[item] = len(gt_cluster_dict)
        else:
            gt_cluster_dict = {}
            for cluster in list(gt_clusters.values()):
                set_cluster = str(cluster)
                if set_cluster not in gt_cluster_dict.keys():
                    gt_cluster_dict[set_cluster] = len(gt_cluster_dict)

    else:
        gt_cluster = pd.Series(gt_clusters.values()).unique()
        gt_cluster_dict = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
    return gt_clusters, ground_t, gt_cluster_dict


def most_frequent_list(nested_list):
    tuples_list = [tuple(sublist) for sublist in nested_list]
    count = Counter(tuples_list)
    most_common_tuple = count.most_common(1)[0][0]
    most_common_list = list(most_common_tuple)
    return most_common_list


def metric_Spee(labels_true, labels_pred):
    MI = metrics.mutual_info_score(labels_true, labels_pred)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    rand_score = metrics.rand_score(labels_true, labels_pred)
    adjusted_random_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    FMI = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    # smc = SMC(labels_true, labels_pred)
    return {"MI": MI, "NMI": NMI, "AMI": AMI, "random Index": rand_score,
            "ARI": adjusted_random_score, "FMI": FMI}


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def rand_Index_custom(predicted_labels, ground_truth_labels):
    print(len(predicted_labels))
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
                all += 1
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

    print(f"TP {true_positive}, TN {true_negative} all {all}")
    RI = (true_positive + true_negative) / all
    return RI


def evaluate_cluster(gtclusters, gtclusters_dict, clusterDict: dict, folder=None,
                     tables_gt: Optional[dict] = None):  # , graph = None
    clusters_label = {}
    table_label_index = []
    false_ones = []

    gt_table_label = []
    tables = []

    overall_info = []
    overall_clustering = []
    ave_consistency = 0

    for index, tables_list in clusterDict.items():

        labels = []
        for table in tables_list:
            if table in gtclusters.keys():
                tables.append(table)
                label = gtclusters[table]
                if type(label) is list:
                    labels.append(label)
                    gt_table_label.append(label)
                else:

                    labels.append([label])
                    gt_table_label.append(gtclusters_dict[label])

        if len(labels) == 0:
            continue
        else:
            cluster_label = most_frequent_list(labels)
            # print(cluster_label)

        clusters_label[index] = cluster_label
        current_ones = []

        for table in tables_list:
            if table in gtclusters.keys():
                if type(cluster_label) is list:
                    table_label_index.append(cluster_label)
                    if len(list(set(gtclusters[table]) & set(cluster_label))) == 0:
                        false_ones.append([table, cluster_label, gtclusters[table]])
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "False", tables_gt[table], gtclusters[table]])
                    else:
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "True", tables_gt[table], gtclusters[table]])
                else:
                    table_label_index.append(gtclusters_dict[cluster_label])
                    if gtclusters[table] != cluster_label:
                        false_ones.append([table, cluster_label, gtclusters[table]])
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "False", tables_gt[table], gtclusters[table]])
                    else:
                        if tables_gt is not None and folder is not None:
                            current_ones.append([table, index, "True", tables_gt[table], gtclusters[table]])
        lowest_gt = [i[3] for i in current_ones]
        # consistency_score = consistency_of_cluster(graph, lowest_gt)
        # ave_consistency += consistency_score
        if tables_gt is not None and folder is not None:
            falses = [i for i in current_ones if i[2] == "False"]
            purity_index = 1 - len(falses) / len(tables_list)
            overall_clustering.extend(current_ones)
            # overall_info.append([index, cluster_label, purity_index, consistency_score, len(tables_list)])
            overall_info.append([index, cluster_label, purity_index, len(tables_list)])

    if type(gt_table_label[0]) is not list:
        metric_dict = metric_Spee(gt_table_label, table_label_index)
    else:
        metric_dict = {"Random Index": rand_Index_custom(gt_table_label, table_label_index)}

    # cb_pairs = wrong_pairs(gt_table_label, table_label_index, tables, tables_gt)
    metric_dict["Purity"] = 1 - len(false_ones) / len(gtclusters)
    # metric_dict["Average cluster consistency score"] = ave_consistency / len(clusterDict)

    if tables_gt is not None and folder is not None:
        df = pd.DataFrame(overall_clustering,
                          columns=['tableName', 'cluster id', 'table type', 'lowest type', 'top level type'])
        df2 = pd.DataFrame(overall_info,
                           columns=['Cluster id', 'Corresponding top level type', 'cCluster purity', 'Size'])
        # 'Consistency of cluster', 'Size'])
        df.to_csv(os.path.join(folder, 'overall_clustering.csv'), encoding='utf-8', index=False)
        df2.to_csv(os.path.join(folder, 'purityCluster.csv'), encoding='utf-8', index=False)
        del df, df2
    return metric_dict



def column_gts(dataset):
    """
     gt_clusters: tablename.colIndex:corresponding label dictionary.
      e.g.{Topclass1: {'T1.a1': 'ColLabel1', 'T1.b1': 'ColLabel2',...}}
    ground_t: label and its tables. e.g.
     {Topclass1:{'ColLabel1': ['T1.a1'], 'ColLabel2': ['T1.b1'', 'T1.c1'],...}}
    gt_cluster_dict: dictionary of index: label
    like  {Topclass1:{'ColLabel1': 0, 'ColLabel2': 1, ...}}
    """
    groundTruth_file = "E:\Project\CurrentDataset\datasets\WDC\column_gt.csv"
    ground_truth_df = pd.read_csv(groundTruth_file, encoding='latin1')
    Superclass = ground_truth_df['TopClass'].dropna().unique()

    gt_clusters = {}
    ground_t = {}
    gt_cluster_dict = {}
    for classTable in Superclass:
        gt_clusters[classTable] = {}
        ground_t[classTable] = {}
        grouped_df = ground_truth_df[ground_truth_df['TopClass'] == classTable]

        for index, row in grouped_df.iterrows():
            gt_clusters[classTable][row["fileName"][0:-4] + "." + str(row["colName"])] = str(row["ColumnLabel"])
            if row["ColumnLabel"] not in ground_t[classTable].keys():
                ground_t[classTable][str(row["ColumnLabel"])] = [row["fileName"][0:-4] + "." + str(row["colName"])]
            else:
                ground_t[classTable][str(row["ColumnLabel"])].append(row["fileName"][0:-4] + "." + str(row["colName"]))
        gt_cluster = pd.Series(gt_clusters[classTable].values()).unique()
        gt_cluster_dict[classTable] = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
    return gt_clusters, ground_t, gt_cluster_dict