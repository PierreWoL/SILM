import json
import os
import pickle
import pandas as pd
from overlap import count_overlapping_nodes
import networkx as nx
# from iterativeFigure import draw_interactive_graph
from utils.folder import mkdir
from utils.post_processing import PostProcessHP, post_process
from Metrics.treeConsistencyScore import updateGroundTruthLabel, TreeConsistencyScore
from Metrics.Topleveltype import topLevelTypeTest
import re

# dataset = "WDC"
# target_path = f"Result/{dataset}/Prompt0/1/deepseek/"
# json_file = os.path.join(target_path, "results.jsonl")

def merge_and_evaluate(dataset, target_path, json_file, test_threshold=0.6, test_Whole = False):
    with open(json_file, 'r', encoding="utf-8") as f:
        data = f.readlines()
    G = nx.DiGraph()
    G.graph["root"] = "Thing"
    max_length = 0
    max_length_path = []
    average, number = 0, 0
    print(json_file)
    for line in data:
        entry = json.loads(line)
        # print(entry)
        table_id = entry["id"]
        hierarchy = entry["paths"]#entry["hierarchy"]
        #paths = hierarchy.strip().split("\n")
        paths = re.findall(r'Thing ->[^\n#]*', hierarchy)

        for path in paths:
            # print(path)
            ###TODO MAYBE NOT START FROM THING
            nodes = path.split(" -> ")[1:]
            average += len(nodes)
            number += 1
            if len(nodes) > max_length:
                max_length = len(nodes)
                max_length_path = nodes
            for i in range(len(nodes) - 1):
                source = nodes[i].lower().capitalize()
                target = nodes[i + 1].lower().capitalize()
                for node in [source, target]:
                    if len(str(node))<=25:
                        if node not in G:
                            G.add_node(node)
                            G.nodes[node]['tables'] = [table_id]
                        else:
                            if 'tables' in G.nodes[node]:
                                G.nodes[node]['tables'].append(table_id)
                            else:
                                G.nodes[node]['tables'] = [table_id]
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    if len(source)<=25 and len(target)<=25:
                        G.add_edge(source, target, weight=1)
    posthp = PostProcessHP()
    G_processed, edges_to_remove_len = post_process(G, posthp)
    with open(os.path.join(target_path, "treeRefine.pkl"), 'wb') as f:
        pickle.dump(G_processed, f)
    print("Top nodes", [i for i in G_processed.nodes if G_processed.in_degree(i) == 0])
    #for u, v, data in G_processed.edges(data=True):
        #print(f"Edge: ({u} -> {v}), Weight: {data['weight']}")
    # print(edges_to_remove_len)
    print(f"max length: {max_length}", max_length_path, f" average length of path is {average / number}")
    print("Nodes:", G_processed.number_of_nodes())
    print("Edges:", G_processed.number_of_edges())
    print(target_path)
    ### TODO: this needs to delete we don't want name matching
    tree = updateGroundTruthLabel(dataset, G_processed)  #
    # draw_interactive_graph( G_processed, os.path.join(target_path, "tree.html"))
    topLevelTypeTest(dataset, tree, output_dir=target_path)

    def find_paths_and_depth(digraph: nx.DiGraph()):
        roots = [node for node in digraph.nodes if digraph.in_degree(node) == 0]
        leaves = [node for node in digraph.nodes if digraph.out_degree(node) == 0]
        all_paths = []
        max_depth = 0
        for root in roots:
            for leaf in leaves:
                if nx.has_path(digraph, root, leaf):
                    path = nx.shortest_path(digraph, source=root, target=leaf)
                    all_paths.append(" -> ".join(path))
                    max_depth = max(max_depth, len(path) - 1)
        print(f"paths number {len(all_paths)}")
        output_file = os.path.join(target_path, "graph_paths.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for path in all_paths:
                f.write(path + "\n")
        return all_paths, max_depth, output_file

    (overall_path_score, all_pathsN, path_tp,
     wrong_nodesN, top_layer, lowestN) = TreeConsistencyScore(tree,
                                                              dataset,
                                                              threshold=test_threshold,
                                                              whole =test_Whole )
    paths, depth, file_path = find_paths_and_depth(tree)
    # print("所有路径：")
    # for path in paths:
    # print(path)
    print(f"\nDiGraph 的最大深度为：{depth}")
    print(f"路径已保存至文件：{file_path}")

    target_path_gt = f"datasets/{dataset}/"
    with open(os.path.join(target_path_gt, "graphGroundTruth.pkl"), "rb") as file:
        G = pickle.load(file)

    overlapping_count, matched_pairs = count_overlapping_nodes(G.nodes(), G_processed.nodes())
    print(overlapping_count, len(G.nodes()), matched_pairs)

    file_path = f"Result/{dataset}/TCSThingRefineSelected.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        columns = ['Method', 'TCS', 'Lowest level types#', 'Top level types#', 'Top level types',
                   'Lowest level types with wrong paths#', 'T(ypes)#',
                   'Wrong paths#', 'Paths#', 'L#', 'Max_length_path#',
                   'Average length of paths', 'Max_length_path']
        df = pd.DataFrame(columns=columns)
    new_row = {'Method': target_path, 'TCS': overall_path_score,
               'Lowest level types#': lowestN, 'Top level types#': len(top_layer), 'Top level types': top_layer,
               'Lowest level types with wrong paths#': wrong_nodesN, 'T(ypes)#': G_processed.number_of_nodes(),
               'Wrong paths#': all_pathsN - path_tp, 'Paths#': all_pathsN, 'L#': depth - 1,
               'Max_length_path#': max_length - 1,
               'Average length of paths': average / number - 1, 'Max_length_path': max_length_path[1:]}
    new_df = pd.DataFrame([new_row])
    print(new_df)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(file_path, index=False)


# 'results', 'results_constraintSTL','results_constraintFET', 'results_constraintFull', 'results_constraintABS'
json_types = [
    'results', 'results_constraintSTL', 'results_constraintFET', 'results_constraintABS', 'results_constraintFull']
dataset_p = "GDS"
model = "qwen14b"  # GPT3
promptchoice =[0]  # '0', '1', '3'
size_choice = [3,5,7,10]
thres = [0.6]
"""for index,json_type in enumerate(json_types):
        for thre in thres:
            target_p = f"Result/{dataset_p}/Step1/Selected/Prompt{index}/0/{model}/"  # /refine
            mkdir(target_p)
            json_file_p = os.path.join(f"Result/{dataset_p}/Step1/Selected/Prompt{index}/0/{model}/", f"{json_type}.jsonl")
            merge_and_evaluate(dataset_p, target_p, json_file_p, test_threshold=thre, test_Whole=False)
"""
for sizec in size_choice:
    target_p = f"Result/{dataset_p}/Step1/Selected/Prompt0/0/{model}/{sizec}/"  # /refine
    mkdir(target_p)
    json_file_p = os.path.join(f"Result/{dataset_p}/Step1/Selected/Prompt0/0/{model}/{sizec}/", f"results.jsonl")
    merge_and_evaluate(dataset_p, target_p, json_file_p, test_threshold=0.5, test_Whole=False)