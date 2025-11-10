import pickle
import pandas as pd
from utils.folder import mkdir
import os
from Metrics.groundTruth import data_classes, evaluate_cluster


def topLevelTypeTest(dataset, tree, output_dir=None):

    groundTruth = f"datasets/{dataset}/groundTruthSelected.csv"
    dataPath = f"datasets/{dataset}/Test/"
    if output_dir is not None:
        mkdir(output_dir)
    top_level_children = list(tree.successors("Thing"))
    print(f"top level type # {len(top_level_children)} {top_level_children}")
    child_clusters = {child: list(set(tree.nodes[child]["tables"])) for i, child in enumerate(top_level_children)}
    gt_clusters, ground_t, gt_cluster_dict = data_classes(dataPath, groundTruth)
    gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(dataPath, groundTruth, superclass=False)
    metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, child_clusters, output_dir,
                                     gt_clusters0)

    new_row = {"Method": output_dir, "Rand Index": metrics_value["Random Index"], "Purity":metrics_value["Purity"]}
    mkdir(f"Result/{dataset}/")
    file_path = f"Result/{dataset}/topLevelTypesSelected.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        columns = ['Method', 'Rand Index', 'Purity']
        df = pd.DataFrame(columns=columns)
    new_df = pd.DataFrame([new_row])
    print(new_df)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(file_path, index=False)

