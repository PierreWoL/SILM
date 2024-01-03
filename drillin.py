import ast
from collections import Counter
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from clustering import evaluate_cluster, data_classes
from d3l.utils.functions import unpickle_python_object


def rank(rank_tuples):
    numbers = set([i[1] for i in rank_tuples])
    sorted_numbers = sorted(numbers, reverse=True)
    dict_rank = {i: [] for i in sorted_numbers}
    for tu in rank_tuples:
        dict_rank[tu[1]].append(tu[0])
    return dict_rank


data_path = "datasets/TabFact/Test/"
groundTruth = "datasets/TabFact/groundTruth.csv"
gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, groundTruth)
gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(data_path, groundTruth, superclass=False)
# print(gt_clusters)
# print(gt_cluster_dict)
# print(gt_clusters0)
tree = unpickle_python_object("datasets/TabFact/graphGroundTruth.pkl")
top = [i for i in tree.nodes() if tree.in_degree(i) ==0]
print(top)
"""
for to in top[-5:-4]:

    successors_of_A = list(tree.successors(to))
    print(to,successors_of_A )
    copytree= tree.copy()
    keep_nodes = nx.descendants(copytree, to)
    deletes = copytree.nodes() - keep_nodes
    copytree.remove_nodes_from(deletes)

    graph_layout = nx.drawing.nx_agraph.graphviz_layout(copytree, prog="dot", args="-Grankdir=TB")
    plt.figure(figsize=(35, 35))
    nx.draw(copytree, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
    plt.savefig(f"datasets/TabFact/{to}.png")
    plt.show()
"""


subject_Col = ['Pretrain_sbert_head_column_none_False', 'Pretrain_sbert_head_column_header_False']

all = ['cl_SC5_lm_sbert_head_column_0_header','cl_SC5_lm_sbert_head_column_0_none',
       'cl_SCT5_lm_sbert_head_column_0_header','cl_SCT5_lm_sbert_head_column_0_none',
       'Pretrain_sbert_head_column_none_False', 'Pretrain_roberta_head_column_header_False'] #_subCol

"""mid = 'All'
for all_file in all:
    result_csv = pd.read_csv(
        f"result/starmie/TabFact/{mid}/OLD/{all_file}/overall_clustering.csv")
    gt = pd.read_csv("datasets/TabFact/groundTruth.csv")
    ids = result_csv["cluster id"].unique()
    cluster_dict = {i: [] for i in ids}
    overall_info = []
    for index, row in result_csv.iterrows():
        cluster_dict[row["cluster id"]].append(row["tableName"])  # +".csv"
    # print(cluster_dict)

    metric_dict = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict,
                                   f"result/starmie/TabFact/{mid}/OLD/{all_file}/",
                                   gt_clusters0)
    metric_df = pd.DataFrame([metric_dict])
    metric_df.to_csv(f"result/starmie/TabFact/{mid}/OLD/{all_file}_metrics.csv")
    print(metric_dict)"""



"""
for id in ids:
    appear_labels = []
    tops = []
    for table in cluster_dict[id]:
        select = gt[gt["fileName"] == table + ".csv"]["class"]
        if not pd.isnull(select).all():
            lt = ast.literal_eval(list(gt[gt["fileName"] == table + ".csv"]["class"])[0])
            ances = []
            for type in lt:
                anc = nx.ancestors(tree, type)
                topan = [i for i in anc if tree.in_degree(i) == 0]
                anc = [i for i in anc if tree.in_degree(i) != 0]
                for i in anc:
                    if i not in ances:
                        ances.append(i)
                tops.extend(topan)
            appear_labels.extend(ances)
    frequency = Counter(appear_labels).most_common()
    ranks = rank(frequency)
    top = Counter(tops).most_common()[0][0]

    overall_info.append([id, top, list(ranks.values())[0], len(cluster_dict[id])])
    print(id,top, ranks)
df2 = pd.DataFrame(overall_info, columns=['Cluster id', 'top level type', 'most frequently appear type', 'Size'])"""
#df2.to_csv("result/starmie/TabFact/Subject_Col/Pretrain_sbert_head_column_none_False/overall_clusteringNew.csv")
# metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, cluster_dict, folder="result/starmie/TabFact/All/cl_SCT6_lm_sbert_head_column_0_header_subCol/",tables_gt=gt_clusters0)
# print(metrics_value)

"""
column_based = ['cl_SC6_lm_sbert_head_column_0_header_column','cl_SC6_lm_sbert_head_column_0_none_column',
       'cl_SCT6_lm_sbert_head_column_0_header_column','cl_SCT6_lm_sbert_head_column_0_none_column'] #_subCol
for i in column_based:
    print(i)
    embedding_i = unpickle_python_object(f"result/embedding/TabFact/{i}.pkl")
    print(embedding_i)
    break
    
"""




