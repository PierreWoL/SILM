import json
import os
import pickle
import networkx as nx
import pandas as pd
from utils.folder import mkdir
from utils.post_processing import PostProcessHP, post_process
from Metrics.treeConsistencyScore import updateGroundTruthColumnLabel
from CA_evaluation import column_gts, evaluate_col_cluster
from collections import defaultdict
import re


def build_column_to_types(G: nx.DiGraph):
    """

    Parameters
    ----------
    G: semantic type hierarchy

    Returns
    -------

    """
    column_to_types = defaultdict(set)
    for node in G.nodes:
        for col in G.nodes[node].get('columns', []):
            column_to_types[col].add(node)
    print("total column", len(column_to_types))
    return column_to_types


def invert_column_to_type(column_to_type):
    type_to_columns = defaultdict(list)
    for col, typ in column_to_type.items():
        type_to_columns[typ].append(col)
    return dict(type_to_columns)


def get_specific_columns_and_types(G: nx.DiGraph):
    column_to_types = build_column_to_types(G)
    # 1.Identify the specific columns associated with each column type —
    # that is, the columns that are unique to this type
    # and not inherited by any of its descendants.
    type_to_specific_columns = defaultdict(set)
    for t in G.nodes:
        current_cols = set(G.nodes[t].get('columns', []))
        descendants = nx.descendants(G, t)
        descendant_cols = set()
        for d in descendants:
            descendant_cols.update(G.nodes[d].get('columns', []))
        specific_cols = current_cols - descendant_cols
        type_to_specific_columns[t] = specific_cols

    # 2. For each column, identify the most specific type it belongs to in the type hierarchy.
    def get_leaf_types_for_column(column):
        types = column_to_types[column]
        leafTypes = set()
        for t in types:
            descens = nx.descendants(G, t)
            if not any(d_type in types for d_type in descens):
                leafTypes.add(t)
        return leafTypes
    def find_deepest_unbranched_common_ancestor(nodes):
        if not nodes:
            return None
        ancestor_sets = [nx.ancestors(G, n) | {n} for n in nodes]
        common_ancestors = set.intersection(*ancestor_sets)

        if not common_ancestors:
            return None

        def is_unbranched_path_to(node):
            """判断从 root 到 node 的路径是否在每一步都只有一个后继"""
            try:
                path = nx.shortest_path(G, source="Thing", target=node)
            except nx.NetworkXNoPath:
                return False
            for i in range(len(path) - 1):
                succ = list(G.successors(path[i]))
                if len(succ) != 1:
                    return False
            return True
        unbranched_common_ancestors = [
            n for n in common_ancestors if is_unbranched_path_to(n)
        ]
        if not unbranched_common_ancestors:
            return None

        return max(unbranched_common_ancestors, key=lambda x: nx.shortest_path_length(G, source="Thing", target=x))

    column_to_specific_type = dict()
    for col in column_to_types:
        leaf_types = get_leaf_types_for_column(col)
        if len(leaf_types) == 1:
                only_leaf = next(iter(leaf_types))
                # 找 root（假设只有一个入度为 0 的节点）
                roots = [n for n in G.nodes if G.in_degree(n) == 0]
                root = roots[0]  # 如有多个根，可以加逻辑选定或报错
                path = nx.shortest_path(G, source=root, target=only_leaf)
                ###TODO this needs refine

                column_to_specific_type[col] = only_leaf
            #except (nx.NetworkXNoPath, IndexError):
                # fallback：出错时保守选择 leaf 本身
                #column_to_specific_type[col] = only_leaf
        else:
            #lca = find_lca_of_nodes(leaf_types)
            lca = find_deepest_unbranched_common_ancestor(leaf_types)
            column_to_specific_type[col] = lca
    print("recheck",len(column_to_specific_type.keys()))
    return type_to_specific_columns, column_to_specific_type


def find_nodes_with_all_columns_and_group_by_label(graph: nx.DiGraph):
    # Step 1: Collect all columns in the dataset,
    # using their names as unique identifiers.
    all_columns = set()
    for node in graph:
        label = graph.nodes[node].get("columns", [])
        for col in graph.nodes[node].get("columns", []):
            if col not in all_columns:
                all_columns.add(col)

    # Step 2: Identify the nodes that cover all columns in the full set.
    qualified_nodes = []
    for node, data in graph.nodes(data=True):
        node_cols = {col["name"] for col in data.get("columns", [])}
        if all_columns.issubset(node_cols):
            qualified_nodes.append((node, data))

    # Step 3: Group the nodes based on the labels of their associated columns.
    grouped_nodes = defaultdict(list)  # label -> list of node ids

    for node_id, data in qualified_nodes:
        labels = set(col["label"] for col in data["columns"])
        for label in labels:
            grouped_nodes[label].append(node_id)

    return grouped_nodes


def merge(dataset, target_path, json_file):
    with open(json_file, 'r', encoding="utf-8") as f:
        data = f.readlines()
    cols_hierarchy = {}
    for line in data:
        entry = json.loads(line)
        table_id = entry["id"]
        attrs_dict = entry["attrs"]
        for dict_path in attrs_dict:
            col_name = dict_path["column"]
            path = dict_path.get("value", dict_path.get("paths"))

            if path is None:
                continue
            #paths = path.strip().split("\n")
            paths = re.findall(r'Thing ->[^\n#]*', path)
            print(paths)

            cols_hierarchy[f"{table_id}.{col_name}"] = paths
    G = nx.DiGraph()
    G.graph["root"] = "Thing"
    max_length = 0
    max_length_path = []
    average, number = 0, 0
    print(json_file)
    for col_name, paths in cols_hierarchy.items():
        for path in paths:
            print("Path ",path)
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
                    if len(node)>50:
                        continue
                    if node not in G:
                        G.add_node(node)
                        G.nodes[node]['columns'] = [col_name]
                    else:
                        if 'columns' in G.nodes[node]:
                            G.nodes[node]['columns'].append(col_name)
                        else:
                            G.nodes[node]['columns'] = [col_name]
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    # if source != target:
                    G.add_edge(source, target, weight=1)
    posthp = PostProcessHP()
    G_processed, edges_to_remove_len = post_process(G, posthp)
    with open(os.path.join(target_path, "treeRefine.pkl"), 'wb') as f:
        pickle.dump(G_processed, f)
    print("Top nodes", [i for i in G_processed.nodes if G_processed.in_degree(i) == 0], average , number)
    #for u, v, data in G_processed.edges(data=True):
        #print(f"Edge: ({u} -> {v}), Weight: {data['weight']}")
    print(edges_to_remove_len)
    print(f"max length: {max_length}", max_length_path, f" average length of path is {average / number}")
    print("Nodes:", G_processed.number_of_nodes())
    print("Edges:", G_processed.number_of_edges())
    print(target_path)
    tree = updateGroundTruthColumnLabel(dataset, G_processed)  #
    print("update GT finish")
    type_to_columns, column_to_type = get_specific_columns_and_types(tree)
    print("Update columns and its belonging types finished")
    result_dict = invert_column_to_type(column_to_type)
    count = 0
    for k, v in result_dict.items():
        count+=len(v)
    print("constructed dict")

    def find_paths_and_depth(digraph):
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
            for p in all_paths:
                f.write(p + "\n")
        return all_paths, max_depth, output_file

    paths, depth, file_path = find_paths_and_depth(tree)
    print(f"\nDiGraph 的最大深度为：{depth}")
    print(f"路径已保存至文件：{file_path}")
    target_f = f"Result/{dataset}/Step2/Prompt0/0/{model}/hierarchyExample/"
    os.makedirs(target_f, exist_ok=True)

    def findTables():
        reference_path = f"datasets/{dataset}/"
        gt_csv = pd.read_csv(os.path.join(reference_path, "groundTruth.csv"))
        class_to_files = gt_csv.groupby('superclass')['fileName'].apply(set).to_dict()
        return class_to_files

    class_file_Dict = findTables()

    def split_column_grouping_by_table_group(table_grouping, column_grouping):
        result = {}
        total = 0
        for group_name, tables in table_grouping.items():
            grouped_columns = defaultdict(list)
            for attr, columns in column_grouping.items():
                for col in columns:
                    prefix = col.split('.')[0]+".csv"  # 提取表名前缀如 '1.csv'
                    if prefix in tables:
                        simplified_col = col.replace('.csv.', '.')
                        grouped_columns[attr].append(simplified_col)
            check_per = 0
            for k,v in grouped_columns.items():
                check_per+=len(v)
            total+=check_per
            result[group_name] = dict(grouped_columns)
        print("rererecheck",total)
        return result

    fet_dict = split_column_grouping_by_table_group(class_file_Dict,result_dict)


    #test_cols(dataset, result_dict, superClass=False,folder=target_f)
    gt_clusters, ground_t, gt_cluster_dict = column_gts(dataset, superclass=True)
    # print(gt_clusters, ground_t, gt_cluster_dict )
    results = []
    delete_id = []
    for fet, fet_result in fet_dict.items():
        fet_test_metrics = evaluate_col_cluster(gt_clusters[fet], gt_cluster_dict[fet],
                                                fet_dict[fet], folder=target_f, fet=fet)
        print(fet_test_metrics)
        if fet_test_metrics is not None:
            results.append(fet_test_metrics)
        else:
            delete_id.append(fet)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(target_path, "result.csv"))


# 'results', 'results_constraintSTL','results_constraintFET', 'results_constraintFull', 'results_constraintABS'
json_types = ['results']
dataset_p = "WDC"
model = "gpt3.5"  # GPT3
for json_type in json_types:
    target_p = f"Result/{dataset_p}/Step2/Prompt0/0/{model}/{json_type}/"  # /refine
    target_p = f"Result/{dataset_p}/Step2/Prompt0/0/{model}/"
    mkdir(target_p)
    json_file_p = os.path.join(f"Result/{dataset_p}/Step2/Prompt0/0/{model}/", f"{json_type}.jsonl")
    merge(dataset_p, target_p, json_file_p)
