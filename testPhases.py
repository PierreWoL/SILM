import os
import pickle
import time

import pandas as pd
from Unicorn.unicorn.model.encoder import DebertaBaseEncoder
from Unicorn.unicorn.model.matcher import MOEClassifier
from Unicorn.unicorn.model.moe import MoEModule
from transformers import DebertaTokenizer, AutoTokenizer

from Unicorn.unicorn.trainer import evaluate
from Unicorn.unicorn.utils.utils import get_data, init_model
from Unicorn.unicorn.dataprocess import predata, dataformat

limit = 10000
import csv
import argparse

csv.field_size_limit(500 * 1024 * 1024)
from Unicorn.unicorn.utils import param


def parse_arguments():
    # argument parsing

    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/moe/classifier')
    parser.add_argument('--dataset', type=str, default="WDC",
                        help="chosen dataset")
    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bert",
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

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
    parser.add_argument('--modelname', type=str, default="UnicornZeroTemp",# UnicornPlus
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


from Utils import subjectColDetection


def transfromCol(column: pd.Series):
    text = "[ATT] " + str(column.name)
    for cell in column:
        text += f" [VAL] {str(cell)}"
    return text



def read_column_corpus(dataset, isSubCol=False, rest=False, max_token=255)->dict:
    def cut(tokenizer, sentence):
        encoded_input = tokenizer(sentence)
        token_count = len(encoded_input['input_ids'])
        if token_count > max_token:
            sentence = tokenizer.decode(encoded_input[:max_token], skip_special_tokens=True)
        return sentence

    table_dict = {}
    SE = None
    dataPath = os.path.join("datasets", dataset, "Test")
    resultPath = os.path.join("datasets", dataset)
    table_names = [i for i in os.listdir(dataPath) if i.endswith(".csv")]
    SE = subjectColDetection(dataPath, resultPath)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    for table_name in table_names:
        table = pd.read_csv(os.path.join(dataPath, table_name))
        annotation, NE_column_score = SE[table_name]
        sub_index = -1
        rest_index = range(len(table.columns))
        if NE_column_score:
            max_score = max(NE_column_score.values())
            sub_index = [key for key, value in NE_column_score.items() if value == max_score][0]
        if isSubCol is True:
            if sub_index != -1:
                column_store = transfromCol(table.iloc[:,sub_index])
                # column_store = cut(tokenizer,column_store)
                table_dict[table_name] = column_store
                # print(column_store)
        elif rest is True:
            table_dict[table_name] = []
            rest_index = [i for i in rest_index if i !=sub_index]
            for index in rest_index:
                column_store = transfromCol(table.iloc[:, index])
                col_name = table.columns[index]
                # column_store = cut(tokenizer, column_store)
                table_dict[f"{table_name}||{col_name}"] = column_store
    return  table_dict


def find_clusters(data_pairs):
    from collections import defaultdict

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

    return clusters
def phase1(encoder, moelayer, classifiers,dataset):
    start_time_encode = time.time()
    corpus = read_column_corpus(dataset, isSubCol=True)

    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    table_names = list(corpus.keys())
    test_data_loaders =[]
    pairs = []

    table_pairs = []
    for index, name in enumerate(table_names):
        for other_name in table_names[index+1:]:
            table_pairs.append((name, other_name))
            pairs.append([corpus[name]+ " [SEP] " + corpus[other_name]])
    fea = predata.convert_examples_to_features(pairs,  max_seq_length=args.max_seq_length, tokenizer = tokenizer)
    test_data_loaders.append(predata.convert_fea_to_tensor(fea, 32, do_train=0))

    predicts = evaluate.matchingOutput(encoder, moelayer, classifiers, test_data_loaders[0], args=args)

    end_time_encode = time.time()
    elapsed_time_encode = end_time_encode - start_time_encode
    # test print
    print("pairs length",len(predicts) ,len(table_pairs))
    start_time_cluster = time.time()
    data_pairs = [(table_pairs[index][0], table_pairs[index][1], predicts[index]) for index in range(len(table_pairs))]
    clusters = find_clusters(data_pairs)
    end_time_cluster = time.time()
    elapsed_time_cluster = end_time_cluster - start_time_cluster
    print(clusters)
    print("encode time",elapsed_time_encode )
    print("cluster time", elapsed_time_cluster)



def main():
    # argument setting
    print("=== Argument Setting ===")
    print("experts", args.expertsnum)
    print("encoder: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("epochs: " + str(args.pre_epochs))

    encoder = DebertaBaseEncoder()
    classifiers = MOEClassifier(args.units)
    exp = args.expertsnum
    moelayer = MoEModule(args.size_output, args.units, exp, load_balance=args.load_balance)


    encoder = init_model(args, encoder, restore= args.modelname  + "_" + param.encoder_path)
    classifiers = init_model(args, classifiers, restore=args.modelname + "_" + param.cls_path)
    moelayer = init_model(args, moelayer, restore=args.modelname + "_" + param.moe_path)

    corpus = read_column_corpus(args.dataset, isSubCol=True)
    with open(os.path.join(f"datasets/{args.dataset}/", f'UnicornP1Test_data_loaders{args.dataset}.pickle'), 'rb') as f:
        test_data_loaders = pickle.load(f)
    print("load complete ...")
    table_pairs = []
    table_names = list(corpus.keys())
    for index, name in enumerate(table_names):
        for other_name in table_names[index + 1:]:
            table_pairs.append((name, other_name))

    predicts = evaluate.matchingOutput(encoder, moelayer, classifiers, test_data_loaders[0], args=args)
    data_pairs = [(table_pairs[index][0], table_pairs[index][1], predicts[index].item()) for index in range(len(table_pairs))]
    for pair in data_pairs:
        print(pair)
    with open(os.path.join(f"datasets/{args.dataset}/", f'UnicornP1Result{args.dataset}.pickle'), 'wb') as f:
         pickle.dump(data_pairs, f)
    clusters = find_clusters(data_pairs)
    print(clusters)
    #phase1(encoder, moelayer, classifiers, args.dataset)




if __name__ == '__main__':
    main()
