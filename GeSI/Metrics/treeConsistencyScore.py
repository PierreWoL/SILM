import ast
import os
import pickle
import networkx as nx
import pandas as pd
from collections import Counter
import re
import torch
import subprocess


def get_table_properties_and_descendants(G, node):
    descendants = nx.descendants(G, node)
    descendants.add(node)
    table_properties = []
    for n in descendants:
        table_property = G.nodes[n].get('tables', [])
        table_properties.extend(table_property)
    return table_properties


def findnodeGTTypes(tree, dataset, node, isTop=True):
    names = []
    groundTruth = f"datasets/{dataset}/groundTruthSelected.csv"
    csvgt = pd.read_csv(groundTruth)
    tables = list(set(get_table_properties_and_descendants(tree, node)))

    def get_class_by_filename(file_name):
        # Find the row that matches the given fileName and return its class
        result = csvgt[csvgt['fileName'] == file_name]
        if not result.empty:
            if isTop is True:
                list_results = result.iloc[0]['superclass']
                list_from_str = ast.literal_eval(list_results)
                return list_from_str
            else:
                return result.iloc[0]['class']
        else:
            return None

    name_dict = {}
    for table in tables:
        name = get_class_by_filename(table)
        name_dict[table] = name
        #
        if name is not None:
            if isinstance(name, list):
                names.extend(name)
            else:
                names.append(name)
    counter = Counter(names)

    if len(counter) != 0:
        max_count = max(counter.values())
        most_frequent_elements = [element for element, count in counter.items() if count == max_count]
    else:
        # print(counter, names)
        most_frequent_elements = []
    return most_frequent_elements  # , tp


def judgePath(leaf_nodes, top_nodes, G):
    has_path = False
    for node_l in leaf_nodes:
        for node_t in top_nodes:
            if node_l == node_t:
                has_path = True
            else:
                is_reachable = nx.has_path(G, source=node_t, target=node_l)
                if is_reachable:
                    has_path = True
    return has_path


def get_max_matched_gt(nestedList, G: nx.DiGraph()):
    maxMatchedGT = 0
    if len(nestedList) <= 2:
        maxMatchedGT = len(nestedList)
    else:
        intersection = set(nestedList[0]) & set(nestedList[-1])
        if intersection:
            maxMatchedGT = 2
            for node in nestedList[1:-1]:
                intersection_inter = set(node) & intersection
                if intersection_inter:
                    maxMatchedGT += 1
        else:
            all_paths = []
            for typeT in nestedList[0]:
                for typeL in nestedList[-1]:
                    paths2 = list(nx.all_simple_paths(G, source=typeT, target=typeL))
                    all_paths.extend(paths2)
            for candidate in all_paths:
                matchedGTElement = 2
                innerNestedList = nestedList[1:-1]
                for elem in innerNestedList:
                    if set(elem) & set(candidate):
                        matchedGTElement += 1
                if matchedGTElement > maxMatchedGT:
                    maxMatchedGT = matchedGTElement
    # print(nestedList, maxMatchedGT)
    return maxMatchedGT


def get_max_matched(nestedList, G: nx.DiGraph()):
    def matchedElementCount(nested_matchList, end_num):
        maxMatchedNum = 2
        if end_num == 1:
            return maxMatchedNum
        all_paths = []
        for typeT in nested_matchList[0]:
            for typeL in nested_matchList[end_num]:
                paths2 = list(nx.all_simple_paths(G, source=typeT, target=typeL))
                all_paths.extend(paths2)
        for candidate in all_paths:
            matchedGTElement = 2
            innerNestedList = nested_matchList[1:end_num]
            for elem in innerNestedList:
                if set(elem) & set(candidate):
                    matchedGTElement += 1
            if matchedGTElement > maxMatchedNum:
                maxMatchedNum = matchedGTElement
        return maxMatchedNum

    maxMatchedGT = 0
    if len(nestedList) == 1:
        maxMatchedGT = len(nestedList)
    else:
        for num in range(len(nestedList) - 1, 0, -1):
            has_path = judgePath(nestedList[num], nestedList[0], G)
            if has_path:
                maxMatchedGT = matchedElementCount(nestedList, num)
                # print(nestedList[num], nestedList[0], has_path)
                break
    return maxMatchedGT


from sentence_transformers import SentenceTransformer, util
import torch


def get_least_used_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if result.returncode != 0:
            print("Failed to query nvidia-smi, fallback to CPU.")
            return 'cpu'

        memory_usages = [int(x) for x in result.stdout.strip().split('\n')]
        least_used_gpu = memory_usages.index(min(memory_usages))

        print(f"Using GPU {least_used_gpu} with {memory_usages[least_used_gpu]} MiB memory used.")
        return f'cuda:{least_used_gpu}'

    except Exception as e:
        print(f"Error while selecting GPU: {e}")
        return 'cpu'



def pathMatch(path,
              G2: nx.DiGraph(),
              model,
              threshold=0.7,
              top_k=5,
              G2_nodes_embedding=None):
    def match_node_by_semantic_similarity(node, g2_embeddings=None):
        query_emb = model.encode(node, convert_to_tensor=True)
        g2_nodes = list(G2.nodes)
        if g2_embeddings is None:
            g2_embeddings = model.encode(g2_nodes, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, g2_embeddings)[0]
        topk_indices = torch.topk(cos_scores, k=min(top_k, len(g2_nodes))).indices
        results = []
        for idx in topk_indices:
            score = cos_scores[idx].item()
            if score >= threshold:
                node = g2_nodes[idx]
                results.append(node)
        return results

    available_paths = []
    for n in path:
        result_nodes = match_node_by_semantic_similarity(n, G2_nodes_embedding)
        available_paths.append(result_nodes)
    matchedPathNum = 0
    matchedGT = get_max_matched(available_paths, G2)
    if matchedGT != 0:
        matchedPathNum += 1
        matchedElement = len(available_paths)
        perConsistencyS = matchedGT / matchedElement
    else:
        perConsistencyS = 0
        print(f"Wrong path!", f'inferred {path}\n', f'gt {available_paths} ')

    return perConsistencyS

def pathTableMatch(path, G1, G2):
    available_paths = []
    for n in path:
        result_nodes = G1.nodes[n].get('label', [])
        available_paths.append(result_nodes)
    matchedPathNum = 0
    #print(available_paths)
    hasPath = judgePath(available_paths[-1], available_paths[0], G2)

    if hasPath is True:
        matchedPathNum += 1
        matchedElement = len(available_paths)
        matchedGT = get_max_matched_gt(available_paths[0:], G2)
        perConsistencyS = matchedGT / matchedElement
    else:
        perConsistencyS = 0
        print(f"Wrong path!", f'inferred {path}', f'gt {available_paths}')

    return perConsistencyS

def TreeConsistencyScore(tree, dataset, threshold=0.6, whole=False):
    overall_path_score = 0
    target_path = f"datasets/{dataset}/"
    if whole is True:
        with open(os.path.join(target_path, "wholeGroundTruth.pkl"), "rb") as file:
            groundTruthTree = pickle.load(file)
    else:
        with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
            groundTruthTree = pickle.load(file)
    all_paths = []
    path_tp = 0
    top_layer = list(tree.successors("Thing"))
    print(len(top_layer), top_layer)
    lowest_layer = [node for node in tree.nodes if tree.out_degree(node) == 0]
    # print("top_layer", top_layer, len(top_layer))
    for top_node in top_layer:
        for bottom_node in lowest_layer:
            if nx.has_path(tree, top_node, bottom_node):
                paths = nx.shortest_path(tree, source=top_node, target=bottom_node)
                # print(paths)
                all_paths.append(paths)

    wrong_nodes = []
    print(f"# Lowest level type,{len(lowest_layer)} # all paths {len(all_paths)}")
    """
    model_name = 'all-MiniLM-L6-v2'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    device = get_least_used_gpu()
    model = SentenceTransformer(model_name)
    #model = model.to(device)  # 把模型移到 GPU 上
    """
    tree_nodes = list(groundTruthTree.nodes)
    print("Ground truth nodes", tree_nodes)
    # tree_nodes_embeddings = model.encode(tree_nodes, convert_to_tensor=True)
    for i, path in enumerate(all_paths):
        pathCS = pathTableMatch(path,tree, groundTruthTree )
        if pathCS != 0:
            overall_path_score += pathCS
            path_tp+=1
    overall_path_score = overall_path_score / len(all_paths) if len(all_paths) > 0 else 1
    print(f"Overall path score: {overall_path_score},#of TP path is, {path_tp}, #total paths :{len(all_paths)}")
    return overall_path_score, len(all_paths), path_tp, len(wrong_nodes), top_layer, len(lowest_layer)


def updateGroundTruthLabel(dataset, tree):
    children = list(tree.successors("Thing"))
    lowest_layer = [node for node in tree.nodes if tree.out_degree(node) == 0]
    for child in children:
        tlt = findnodeGTTypes(tree, dataset, child, isTop=True)
        # print(child,tlt )
        tree.nodes[child]['label'] = tlt
    for node in tree.nodes:
        if node not in children:
            llt = findnodeGTTypes(tree, dataset, node, isTop=False)
            tree.nodes[node]['label'] = llt
            #if node in lowest_layer:
               # print(node, llt)
    return tree


def get_cols_and_descendants(G, node):
    descendants = nx.descendants(G, node)
    descendants.add(node)
    col_properties = []
    for n in descendants:
        table_property = G.nodes[n].get('columns', [])
        col_properties.extend(table_property)
    return col_properties


def findnodeGTColTypes(tree, dataset, node):
    names = []
    groundTruth = f"datasets/{dataset}/column_gt.csv"
    csvgt = pd.read_csv(groundTruth, encoding="latin1")
    columns = list(set(get_cols_and_descendants(tree, node)))

    def get_class_by_filename(column_agg):
        # Find the row that matches the given fileName and return its class
        fileName, col_Name = column_agg.rsplit('.', 1)
        result = csvgt[(csvgt['fileName'] == fileName) & (csvgt['colName'] == col_Name)]
        if not result.empty:
            return result.iloc[0]['ColumnLabel']
        else:
            return None

    name_dict = {}
    for col in columns:
        name = get_class_by_filename(col)
        name_dict[col] = name
        if name is not None:
            if isinstance(name, list):
                names.extend(name)
            else:
                names.append(name)
    counter = Counter(names)

    if len(counter) != 0:
        max_count = max(counter.values())
        most_frequent_elements = [element for element, count in counter.items() if count == max_count]
    else:
        # print(counter, names)
        most_frequent_elements = []
    return most_frequent_elements  # , tp


def updateGroundTruthColumnLabel(dataset, tree):
    for node in tree.nodes:
        if node != "Thing":
            llt = findnodeGTColTypes(tree, dataset, node)
            print(node, llt)
            tree.nodes[node]['label'] = llt
    return tree
