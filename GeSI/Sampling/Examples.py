import os
import pickle
import random

from Sampling.Table import example_data
import networkx as nx
import pandas as pd


def sampleFEET(k=3, dataset="WDC"):
    examples = []
    if k == 0:
        return examples
    reference_path = f"E:/Project/datasets/{dataset}/"
    gt_csv = os.path.join(reference_path, "groundTruth.csv")
    gt = pd.read_csv(gt_csv)
    unique_classes = gt['class'].unique()
    if len(unique_classes) < k:
        selected_classes = unique_classes
    else:
        selected_classes = random.sample(unique_classes.tolist(), k)
    print(gt, unique_classes.tolist(), selected_classes)
    for cls in selected_classes:
        row = gt[gt['class'] == cls].sample(1)
        table = example_data(pd.read_csv(os.path.join(reference_path, "Test", str(row["fileName"].values[0]))))
        examples.append({'table': table, 'type': row["class"].values[0]})
    return examples


def sample_examples(n=3, dataset="WDC"):
    examples = []
    if n == 0:
        return examples
    reference_path =f"E:/Project/datasets/{dataset}/"
    gt_csv = os.path.join(reference_path, "groundTruth.csv")
    gt = pd.read_csv(gt_csv)
    unique_classes = gt['class'].unique()
    random_rows = []
    if len(unique_classes) < n:
        print(f"Not enough unique classes to select {n} distinct rows.")
    else:
        selected_classes = random.sample(unique_classes.tolist(), n)
        print(gt, unique_classes.tolist(), selected_classes)
        for cls in selected_classes:
            row = gt[gt['class'] == cls].sample(1)
            random_rows.append(row)
    print(random_rows)
    result = pd.concat(random_rows)

    result_dict = result.set_index('fileName')['class'].to_dict()
    with open(os.path.join(reference_path, 'graphGroundTruth.pkl'), 'rb') as f:
        graph_data = pickle.load(f)

    for key, value in result_dict.items():
        if value not in graph_data:
            continue
        all_paths_to_target = []
        for root in [node for node, in_degree in graph_data.in_degree() if in_degree == 0]:
            # try:
            paths = list(nx.all_simple_paths(graph_data, source=root, target=value))
            all_paths_to_target.extend(paths)

        # except nx.NetworkXNoPath:
        # continue

        # Print the paths
        # print(f"{key}'s Paths from the root(s) to '{value}':", all_paths_to_target)
        # allpaths = []
        # for path in all_paths_to_target:
        # allpaths.append(" -> ".join(path))
        # print(" -> ".join(path))
        table = example_data(pd.read_csv(os.path.join(reference_path, "Test", key)))
        examples.append({'table': table, 'paths': all_paths_to_target})
        # print(key, all_paths_to_target)
    return examples
