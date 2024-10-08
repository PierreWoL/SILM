import os
import pickle
import re
import time

import pandas as pd
import torch
from ClusterHierarchy.ClusterDecompose import tree_consistency_metric
from ClusterHierarchy.JaccardMetric import JaccardMatrix
from RelationshipSearch.SearchRelationship import relationshipGT
from TableCluster.tableClustering import column_gts
from Unicorn.unicorn.model.encoder import DebertaBaseEncoder
from Unicorn.unicorn.model.matcher import MOEClassifier
from Unicorn.unicorn.model.moe import MoEModule
from transformers import DebertaTokenizer
from Unicorn.unicorn.utils import param
from Unicorn.unicorn.trainer import evaluate
from Unicorn.unicorn.utils.utils import init_model
from Unicorn.unicorn.dataprocess import predata
from clustering import data_classes, evaluate_cluster, evaluate_col_cluster
import csv
import argparse
from Utils import subjectColDetection, mkdir

limit = 10000
csv.field_size_limit(500 * 1024 * 1024)


def pathStore(dataset, phase, name=""):
    path = os.path.join(f"result/{phase}/{dataset}/")
    if name != "":
        path = os.path.join(f"result/{phase}/{dataset}/{name}/")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"create path")
    return path


def parse_arguments():
    # argument parsing

    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/moe/classifier')
    parser.add_argument('--dataset', type=str, default="WDC",
                        help="chosen dataset")
    parser.add_argument('--step', type=str, default="P2",
                        help="chosen PHASE")
    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="deberta_base",
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--ratio', type=float, default=0.3,
                        help="ratio of non-subject attributes")
    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default=10,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--pre_log_step', type=int, default=10,
                        help="Specify log step size for pretrain")

    parser.add_argument('--log_step', type=int, default=10,
                        help="Specify log step size for adaptation")

    parser.add_argument('--c_learning_rate', type=float, default=3e-6,
                        help="Specify lr for training")

    parser.add_argument('--num_cls', type=int, default=5,
                        help="")
    parser.add_argument('--num_tasks', type=int, default=2,
                        help="")

    parser.add_argument('--resample', type=int, default=0,
                        help="")
    parser.add_argument('--modelname', type=str, default="UnicornZeroTemp",  # UnicornPlus
                        help="Specify saved model name")
    parser.add_argument('--ckpt', type=str, default="",
                        help="Specify loaded model name")
    parser.add_argument('--num_data', type=int, default=1000,
                        help="")
    parser.add_argument('--num_k', type=int, default=2,
                        help="")

    parser.add_argument('--scale', type=float, default=20,
                        help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")

    parser.add_argument('--wmoe', type=int, default=1,
                        help="with or without moe")
    parser.add_argument('--expertsnum', type=int, default=15,
                        help="number of experts")
    parser.add_argument('--size_output', type=int, default=768,
                        help="encoder output size")
    parser.add_argument('--units', type=int, default=1024,
                        help="number of hidden")

    parser.add_argument('--shuffle', type=int, default=0, help="")
    parser.add_argument('--load_balance', type=int, default=0, help="")

    return parser.parse_args()


args = parse_arguments()


def find_clusters(data_pairs, dataset=None):
    from collections import defaultdict
    result_set = set()

    def dfs(node, visited, graph, cluster):
        visited.add(node)
        cluster.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, graph, cluster)

    # 构建图
    graph = defaultdict(list)
    for pair in data_pairs:
        if pair[2] == 1:  # 如果 match
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])

    visited = set()
    clusters = []
    for node in graph:
        if node not in visited:
            cluster = []
            dfs(node, visited, graph, cluster)
            clusters.append(cluster)
            if dataset is not None:
                result_set.update(cluster)
        # if len(result_set) > 0:
        # print(len(result_set), result_set)

        """lef_clusters = set(dataset) - result_set
        # print(lef_clusters)
        for table in lef_clusters:
            clusters.append([table])"""
    cluster_dict = {index: [i for i in cluster] for index, cluster in enumerate(clusters)}
    return cluster_dict


def write(target, path, name):
    with open(os.path.join(path, f'{name}_{args.dataset}.pickle'), 'wb') as f:
        pickle.dump(target, f)


def read(path, name):
    with open(os.path.join(path, f'{name}_{args.dataset}.pickle'), 'rb') as f:
        target = pickle.load(f)
    return target


def transfromCol(column: pd.Series):
    text = "[ATT] " + str(column.name)
    for cell in column:
        text += f" [VAL] {str(cell)}"
    return text


def read_column_corpus(dataset, isSubCol=False, rest=False, NE=False, max_token=255, selected_dataset=None) -> dict:
    def cut(tokenizer, sentence):
        encoded_input = tokenizer(sentence)
        token_count = len(encoded_input['input_ids'])
        if token_count > max_token:
            sentence = tokenizer.decode(encoded_input[:max_token], skip_special_tokens=True)
        return sentence

    table_dict = {}
    dataPath = os.path.join("datasets", dataset, "Test")
    resultPath = os.path.join("datasets", dataset)
    table_names = [i for i in os.listdir(dataPath) if i.endswith(".csv")]
    if selected_dataset is not None:
        table_names = [i for i in table_names if i in selected_dataset]
    SE = subjectColDetection(dataPath, resultPath)
    for table_name in table_names:
        table = pd.read_csv(os.path.join(dataPath, table_name))
        annotation, NE_column_score = SE[table_name]
        sub_index = -1
        NE_indexes = []
        rest_index = range(len(table.columns))
        if NE_column_score:
            max_score = max(NE_column_score.values())
            sub_index = [key for key, value in NE_column_score.items() if value == max_score][0]
            NE_indexes = [key for key, value in NE_column_score.items() if value != max_score]
        if isSubCol is True:
            if sub_index != -1:
                column_store = transfromCol(table.iloc[:, sub_index])
                table_dict[table_name[:-4]] = column_store
        elif rest is True:
            if NE is False:
                rest_index = [i for i in rest_index if i != sub_index]
                for index in rest_index:
                    column_store = transfromCol(table.iloc[:, index])
                    col_name = table.columns[index]
                    table_dict[f"{table_name[:-4]}.{col_name}"] = column_store
            else:
                for index in NE_indexes:
                    column_store = transfromCol(table.iloc[:, index])
                    col_name = table.columns[index]
                    table_dict[f"{table_name[:-4]}.{col_name}"] = column_store
    return table_dict


def convert(dataset, phase=1, selected=None, Name=""):
    path = pathStore(dataset, f"P{phase}", Name)
    name_corpus = "corpus"
    start_time_encode = time.time()
    if os.path.exists(os.path.join(path, f'{name_corpus}_{args.dataset}.pickle')):
        corpus = read(path, name_corpus)
        print(f"read corpus of {Name} successfully")
    else:
        if phase == 1:
            corpus = read_column_corpus(dataset, isSubCol=True)
        else:
            if phase == 4:
                corpus = read_column_corpus(dataset, isSubCol=True, selected_dataset=selected)
            else:
                if selected is not None:
                    corpus = read_column_corpus(dataset, rest=True, selected_dataset=selected)
                else:
                    corpus = read_column_corpus(dataset, rest=True)
        write(corpus, path, name_corpus)
        print(f"generate corpus of {Name}.")
    name_loader = "testDataLoader"
    table_names = list(corpus.keys())
    table_pairs = []
    pairs = []
    for index, name in enumerate(table_names):
        for other_name in table_names[index + 1:]:
            table_pairs.append((name, other_name))
            pairs.append([corpus[name] + " [SEP] " + corpus[other_name]])

    if os.path.exists(os.path.join(path, f'{name_loader}_{args.dataset}.pickle')):
        test_data_loaders = read(path, name_loader)
        print(f"read test_data_loaders of {Name} successfully")
    else:
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        test_data_loaders = []
        fea = predata.convert_examples_to_features(pairs, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        test_data_loaders.append(predata.convert_fea_to_tensor(fea, 64, do_train=0))
        print(f"generate test_data_loaders of {Name}.")
        write(test_data_loaders, path, name_loader)
        end_time_convert = time.time()
        elapsed_time_convert = end_time_convert - start_time_encode
        print("convert time", elapsed_time_convert)
    return corpus, test_data_loaders, table_pairs


def clustering(encoder, moelayer, classifiers, dataset, phase=1, selected=None, Name=""):
    corpus, test_data_loaders, table_pairs = convert(dataset, phase=phase, selected=selected, Name=Name)
    print(len(table_pairs))
    start_time_encode = time.time()
    predicts = evaluate.matchingOutput(encoder, moelayer, classifiers, test_data_loaders[0], args=args)
    end_time_encode = time.time()

    elapsed_time_encode = end_time_encode - start_time_encode
    print("encode time", elapsed_time_encode)

    start_time_cluster = time.time()
    data_pairs = [(table_pairs[index][0], table_pairs[index][1], predicts[index]) for index in range(len(table_pairs))]
    # write(data_pairs, pathStore(dataset, f"P{phase}", Name), "dataPairs")
    table_names = list(corpus.keys())
    clusters = find_clusters(data_pairs, table_names)
    print(clusters)
    end_time_cluster = time.time()
    elapsed_time_cluster = end_time_cluster - start_time_cluster
    print("cluster time", elapsed_time_cluster)
    return clusters


def phase1(encoder, moelayer, classifiers, dataset):
    clusters = clustering(encoder, moelayer, classifiers, dataset, )
    data_path = os.getcwd() + f"/datasets/{dataset}/Test/"
    ground_truth = os.getcwd() + f"/datasets/{dataset}/groundTruth.csv"
    gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, ground_truth)
    gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(data_path, ground_truth, superclass=False)
    del ground_t0, gt_cluster_dict0
    # folderName = os.getcwd() + f"/datasets/{dataset}"
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, clusters, None,
                                     gt_clusters0)
    return metrics_value


def phase2(encoder, moelayer, classifiers, dataset, intervalSlice=10, delta=0.15):
    data_path = os.getcwd() + f"/datasets/{dataset}/Test/"
    ground_truth_table = os.getcwd() + f"/datasets/{dataset}/groundTruth.csv"
    Ground_t = data_classes(data_path, ground_truth_table, Nochange=True)[1]

    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset)
    for index, clu in enumerate(list(gt_cluster_dict.keys())):
        #if 'CreativeWork' in clu or 'Event' in clu:
          try:
                print(f"currernt cluster: {clu}", len(Ground_t[clu]))
                tables = [i + ".csv" for i in Ground_t[clu]]
                clusters = clustering(encoder, moelayer, classifiers, dataset, phase=2, selected=tables, Name=clu)
                write(clusters, pathStore(dataset, f"P2", clu), "clusters")
                print(clusters)
                if len(clusters) > 0:
                    metrics_value = evaluate_col_cluster(gt_clusters[clu], gt_cluster_dict[clu], clusters)
                    print(clu, metrics_value)

                    start_time_cluster = time.time()
                    jaccard_score = JaccardMatrix(clusters, data_path)[2]
                    TCS, ALL_path = tree_consistency_metric(tables, jaccard_score, "Unicorn", dataset,
                                                            cluster_name="Connect", Naming=str(index),
                                                            sliceInterval=intervalSlice, delta=delta)[:-1]
                    end_time_cluster = time.time()
                    elapsed_time_cluster = end_time_cluster - start_time_cluster
                    print("P3 time", elapsed_time_cluster)
                    print('Top level type ', clu, 'Tree Consistency Score:', TCS, "#Paths:", ALL_path)
          except:
            print("error!")
        

def phase3(dataset, intervalSlice, delta):
    data_path = os.getcwd() + f"/datasets/{dataset}/Test/"
    ground_truth_table = os.getcwd() + f"/datasets/{dataset}/groundTruth.csv"

    Ground_t = data_classes(data_path, ground_truth_table, Nochange=True)[1]
    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset)
    for index, clu in enumerate(list(gt_cluster_dict.keys())):
        print(f"Now is {clu}")
        tables = [i + ".csv" for i in Ground_t[clu]]
        corpus = read(pathStore(dataset, f"P2", clu), "corpus")
        data_pairs = read(pathStore(dataset, f"P2", clu), "dataPairs")
        colClusters = find_clusters(data_pairs, list(corpus.keys()))
        print("colClusters",colClusters)
        if colClusters !={}:
            metrics_value = evaluate_col_cluster(gt_clusters[clu], gt_cluster_dict[clu], colClusters)
            print(clu, metrics_value)
            jaccard_score = JaccardMatrix(colClusters, data_path)[2]
            TCS, ALL_path = tree_consistency_metric(tables, jaccard_score, "Unicorn", dataset,
                                                    cluster_name="Connect", Naming=str(index),
                                                    sliceInterval=intervalSlice, delta=delta)[:-1]
            print('Top level type ', clu, 'Tree Consistency Score:', TCS, "#Paths:", ALL_path)


def phase4(encoder, moelayer, classifiers, dataset, ratio):
    def conceptualAttriClusters(ts):
        corpus_SC = read_column_corpus(dataset, isSubCol=True, selected_dataset=ts)
        corpus_NE = read_column_corpus(dataset, rest=True, NE=True, selected_dataset=ts)
        NE_dict = gt_clusters[clu_i]
        CNE_clusters = {}
        for attri in corpus_NE.keys():
            if attri in NE_dict.keys():
                cne = NE_dict[attri]
            if cne in CNE_clusters.keys():
                CNE_clusters[cne].append(attri)
            else:
                CNE_clusters[cne] = [attri]
        return corpus_SC, corpus_NE, CNE_clusters

    def attriPair(corpus_SC, corpus_NEo, CNEo_clusters, clu1, clu2):
        pairs = {}
        for cne in CNEo_clusters.keys():
            pairs[cne] = {}
            pairs[cne]['pairs'] = []
            pairs[cne]['table_pairs'] = []

            attris = CNEo_clusters[cne]
            pairs[cne]['cne_num'] = len(attris)
            for attri in attris:
                for subAttri in corpus_SC.keys():
                    pairs[cne]['table_pairs'].append((attri, subAttri))
                    pairs[cne]['pairs'].append([corpus_NEo[attri] + " [SEP] " + corpus_SC[subAttri]])
        path = f"result/P4/{dataset}/{clu1}_{clu2}/"
        mkdir(path)
        write(pairs, path, "attrPairs")
        print(pairs.keys())
        return pairs

    def cluster_pair_loader(pairs_dict, clu1, clu2):
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        test_data_loaders = {}
        results = {}
        print(pairs_dict.keys())
        for cne in list(pairs_dict.keys()):
            cne_pair_dict = pairs_dict[cne]
            pairs = cne_pair_dict['pairs']
            table_pairs = cne_pair_dict['table_pairs']
            results[cne] = {}
            test_data_loaders[cne] = []
            fea = predata.convert_examples_to_features(pairs, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
            test_data_loaders[cne].append(predata.convert_fea_to_tensor(fea, 64, do_train=0))
            predicts_cne = evaluate.matchingOutput(encoder, moelayer, classifiers, test_data_loaders[cne][0], args=args)
            attris = set([table_pairs[predicts_cne.index(i)][0] for i in predicts_cne if i == 1])
            results[cne]['predicts'] = predicts_cne
            results[cne]['ratio']=len(attris) / pairs_dict[cne]['cne_num']
            if len(attris) / pairs_dict[cne]['cne_num'] >= ratio:
                results[cne]['isJoin'] = True
            else:
                results[cne]['isJoin'] = False
        path = f"result/P4/{dataset}/{clu1}_{clu2}/"
        mkdir(path)
        write(test_data_loaders, path, "test_data_loaders")
        write(results, path, f"results_{ratio}")
        return results

    gt_relationship = relationshipGT(args.dataset)
    print(gt_relationship)
    data_path = os.getcwd() + f"/datasets/{dataset}/Test/"
    ground_truth_table = os.getcwd() + f"/datasets/{dataset}/groundTruth.csv"
    Ground_t = data_classes(data_path, ground_truth_table, Nochange=True)[1]
    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset)
    keys = list(gt_cluster_dict.keys())
    for index_i, clu_i in enumerate(keys):
        # if "'Place', 'Organization'" in clu_i:
        tables_i = [i + ".csv" for i in Ground_t[clu_i]]
        corpus_SCi, corpus_NEi, CNEi_clusters = conceptualAttriClusters(tables_i)
        for clu_j in keys[index_i + 1:]:
            # if 'Place' in clu_j:
            tables_j = [i + ".csv" for i in Ground_t[clu_i]]
            print(f"Now is {clu_i} nad {clu_j}")
            corpus_SCj, corpus_NEj, CNEj_clusters = conceptualAttriClusters(tables_j)
            CNE_pairs_ij = attriPair(corpus_SCi, corpus_NEj, CNEj_clusters, clu_i, clu_j)
            results_ij = cluster_pair_loader(CNE_pairs_ij, clu_i, clu_j)
            CNE_pairs_ji = attriPair(corpus_SCj, corpus_NEi, CNEi_clusters, clu_j, clu_i)
            results_ji = cluster_pair_loader(CNE_pairs_ji, clu_j, clu_i)
            print(results_ij,results_ji )


def main():
    # argument setting
    print("=== Argument Setting ===")
    print("experts", args.expertsnum)
    print("dataset: ", args.dataset)
    print("encoder: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("epochs: " + str(args.pre_epochs))
    print("ratio: " + str(args.ratio))
    encoder = DebertaBaseEncoder()
    classifiers = MOEClassifier(args.units)
    exp = args.expertsnum
    moelayer = MoEModule(args.size_output, args.units, exp, load_balance=args.load_balance)

    encoder = init_model(args, encoder, restore=args.modelname + "_" + param.encoder_path)
    classifiers = init_model(args, classifiers, restore=args.modelname + "_" + param.cls_path)
    moelayer = init_model(args, moelayer, restore=args.modelname + "_" + param.moe_path)

    if args.step == "P1":
        phase1(encoder, moelayer, classifiers, args.dataset)
    if args.step == "P2":
        phase2(encoder, moelayer, classifiers, args.dataset)
    if args.step == "P3":
        phase3(args.dataset, 10, 0.15)
    if args.step == "P4":
        phase4(encoder, moelayer, classifiers, args.dataset, args.ratio)


if __name__ == '__main__':
    #main()

    path = f"result/P4/{args.dataset}/"
    pair_names = [i for i in os.listdir(path) if '[' in i]
    #print(pair_names)
    gt_relationship = relationshipGT(args.dataset)
    #print(gt_relationship)
    gt_num = len(gt_relationship[1])
    TP = 0
    Detect = 0

    for pair_name in pair_names:
       if "['Place', 'Organization']" in pair_name:
           pair_name2 = pair_name.replace("['Place', 'Organization']", "['Organization']")
           categories = re.findall(r"\['(.*?)'\]", pair_name2)
       else:
           categories = re.findall(r"\['(.*?)'\]", pair_name)
       categories_tuple = tuple(categories)
       NE_relationship_gt = []
       if categories_tuple in gt_relationship:
           NE_relationship_gt = gt_relationship[categories_tuple]
       with open(os.path.join(path, pair_name, "results_WDC.pickle"), 'rb') as f:
           #results_WDC = pickle.load(f)
             results_WDC = torch.load(f, map_location=torch.device('cpu'))
       with open(os.path.join(path, pair_name, "attrPairs_WDC.pickle"), 'rb') as f:
           attrPairs_WDC = pickle.load(f)
       print(results_WDC,attrPairs_WDC )
       for NE_name in results_WDC.keys():
           table_pairs = attrPairs_WDC[NE_name]
           predicts_cne = results_WDC[NE_name]['predicts']
           attris = set([table_pairs['table_pairs'][predicts_cne.index(i)][0] for i in predicts_cne if i == 1])

           if len(attris)>0:
               Detect +=1
               if len(attris) / table_pairs['cne_num'] >= args.ratio:
                   #print(categories_tuple, NE_relationship_gt, NE_name, attris)
                   if NE_name in NE_relationship_gt:
                        TP +=1

    recall = TP/gt_num
    precision = TP/Detect
    print("ratio is ",args.ratio,"recall is :", recall,"Precision is :", precision )
