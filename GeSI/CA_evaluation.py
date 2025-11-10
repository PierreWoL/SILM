import ast
import os
from collections import Counter
import pandas as pd
from sklearn import metrics

def column_gts(dataset, superclass = True):
    """
     gt_clusters: tablename.colIndex:corresponding label dictionary.
      e.g.{Topclass1: {'T1.a1': 'ColLabel1', 'T1.b1': 'ColLabel2',...}}
    ground_t: label and its tables. e.g.
     {Topclass1:{'ColLabel1': ['T1.a1'], 'ColLabel2': ['T1.b1'', 'T1.c1'],...}}
    gt_cluster_dict: dictionary of index: label
    like  {Topclass1:{'ColLabel1': 0, 'ColLabel2': 1, ...}}
    """
    groundTruth_file = os.getcwd() + "/datasets/" + dataset + "/column_gt.csv"
    ground_truth_df = pd.read_csv(groundTruth_file, encoding='latin1')

    Superclass = ground_truth_df['TopClass'].dropna().unique()
    classes =  ground_truth_df['LowestClass'].dropna().unique()

    if superclass:
        check_classes = Superclass
        check_column = 'TopClass'
    else:
        check_classes = classes
        check_column = 'LowestClass'
    gt_clusters = {}
    ground_t = {}
    gt_cluster_dict = {}
    for classTable in check_classes:
        gt_clusters[classTable] = {}
        ground_t[classTable] = {}
        grouped_df = ground_truth_df[ground_truth_df[check_column] == classTable]

        for index, row in grouped_df.iterrows():
            gt_clusters[classTable][row["fileName"][0:-4] + "." + str(row["colName"])] = str(row["ColumnLabel"])
            if row["ColumnLabel"] not in ground_t[classTable].keys():
                ground_t[classTable][str(row["ColumnLabel"])] = [row["fileName"][0:-4] + "." + str(row["colName"])]
            else:
                ground_t[classTable][str(row["ColumnLabel"])].append(row["fileName"][0:-4] + "." + str(row["colName"]))
        gt_cluster = pd.Series(gt_clusters[classTable].values()).unique()
        gt_cluster_dict[classTable] = {cluster: list(gt_cluster).index(cluster) for cluster in gt_cluster}
        # print(len(gt_cluster_dict[classTable]))
    return gt_clusters, ground_t, gt_cluster_dict
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def get_files(data_path):
    T = []
    if data_path.endswith('.csv'):
        features = pd.read_csv(data_path)
        T = features.iloc[:, 0]
    else:
        T = os.listdir(data_path)
        T = [t[:-4] for t in T if t.endswith('.csv')]
        T.sort()
    return (T)


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
    if superclass:
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"), ground_truth_df.iloc[:, 2]))
    else:
        dict_gt = dict(zip(ground_truth_df.iloc[:, 0].str.removesuffix(".csv"), ground_truth_df.iloc[:, 1]))

    # dict_gt = {key: ast.literal_eval(value) for key, value in dict_gt.items() if value != " " and "[" in value}
    dict_gt0 = {}
    for key, value in dict_gt.items():
        if value != " ":
            if "[" in value:
                dict_gt0[key] = ast.literal_eval(value)
            else:
                dict_gt0[key] = value
    dict_gt = dict_gt0
    # print(f"dict_gt {dict_gt}")
    test_table2 = {}.fromkeys(test_table).keys()
    gt_clusters, ground_t = get_concept_files(get_files(data_path), dict_gt, Nochange=Nochange)
    # print(gt_clusters.values())
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


def evaluate_col_cluster(gtclusters, gtclusters_dict, clusterDict: dict, folder=None,fet= None):
    #print(gtclusters, "\n",gtclusters_dict)
    clusters_label = {}
    column_label_index = []
    false_ones = []
    gt_column_label = []
    columns = []
    columns_ref = []
    for index, column_list in clusterDict.items():
        labels = []
        for column in column_list:
            if column in gtclusters.keys():
                columns.append(column)
                label = gtclusters[column]
                if type(label) is list:
                    for item_label in label:
                        labels.append(item_label)
                    gt_column_label.append(label)
                else:
                    gt_column_label.append(gtclusters_dict[label])
                    labels.append(label)
        if len(labels) == 0:
            continue
        else:
            cluster_label = most_frequent(labels)
        clusters_label[index] = cluster_label
        false_cols = []
        for column in column_list:
            if column in gtclusters.keys():
                column_label_index.append(gtclusters_dict[cluster_label])
                if gtclusters[column] != cluster_label:
                    false_cols.append(column)
                    false_ones.append([column,cluster_label,gtclusters[column]])
            columns_ref.append([column_list, cluster_label, false_cols])
        # print(1 - len(false_cols) / len(column_list),len(false_cols), len(column_list))
    # print(gt_column_label,clusterDict)
    if len(gt_column_label) == 0:
        print("Wrong result clusters!")
        return None
    if type(gt_column_label[0]) is not list:
        metric_dict = metric_Spee(gt_column_label, column_label_index)
    else:
        metric_dict = {"random Index": rand_Index_custom(gt_column_label, column_label_index)}
    # cb_pairs = wrong_pairs(gt_table_label, table_label_index, tables, tables_gt)
    metric_dict["purity"] = 1 - len(false_ones) / len(column_label_index)
    if folder is not None and fet is not None:
        df_examples = pd.DataFrame(false_ones, columns=['col name', 'result label', 'true label'])
        df_examples.to_csv(os.path.join(folder, f'{fet}.csv'), encoding='utf-8', index=False)
        # print(df_examples)
    return metric_dict
        #if columns:
            #df_cols = pd.DataFrame(columns_ref, columns=['resultCols', 'result label', 'false_cols'])
            #df_cols.to_csv(os.path.join(folder, 'cols_results.csv'), encoding='utf-8', index=False)



def test_cols(dataset, result_dict, superClass = True, folder=None):
    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset,superclass=superClass)
    #print(gt_clusters, ground_t, gt_cluster_dict )
    results = []
    delete_id = []
    for fet, fet_result in result_dict.items():
        fet_test_metrics = evaluate_col_cluster(gt_clusters[fet], gt_cluster_dict[fet],
                                                result_dict[fet], folder=folder, fet =fet )
        if fet_test_metrics is not None:
            results.append(fet_test_metrics)
        else:
            delete_id.append(fet)

    df = pd.DataFrame(results)
    new_col = list(result_dict.keys())
    left  =[i for i in new_col if i not in delete_id]
    df.insert(0, 'id', left)
    selected_cols = ['purity', 'random Index']
    mean_per_column = df[selected_cols].mean()
    print(mean_per_column)



