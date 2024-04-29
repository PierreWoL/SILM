import pickle
import os
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

import pandas as pd

datasets = ["TabFact"] #,"TabFact"
for dataset in datasets:
    datafile_path = os.path.join(os.getcwd(), "result/embedding/",  dataset)
    gt_filename = "01SourceTables.csv" if  dataset == "TabFact" else "groundTruth.csv"
    ground_truth = os.path.join(os.getcwd(), "datasets", dataset, gt_filename)
    gt = pd.read_csv(ground_truth, encoding='latin1')
    # needs to delete in the future when all labeling is done
    colname = 'superclass' if  dataset == "TabFact" else "Label"
    #tables_with_class = gt[gt[colname].notnull()]
    tables_with_class = gt[gt[colname] == "sports league"]
    #clusters = tables_with_class[colname].unique()
    clusters = tables_with_class["class"].unique()
    print(len(tables_with_class),len(clusters))
    files = [fn for fn in os.listdir(datafile_path) if '.pkl' in fn]
    for file in files:
        print(file)
        F = open(os.path.join(datafile_path, file), 'rb')
        content = pickle.load(F)
        vectors = []
        T = []
        for vector in content:
            T.append(vector[0][:-4])

            vec_table = np.mean(vector[1], axis=0)
            vectors.append(vec_table)
        vectors = np.array(vectors)
        # Perform hierarchical clustering
        Z = linkage(vectors, method='complete')  # You can choose different linkage methods

        # Plot dendrogram
        plt.figure(figsize=(10, 5))
        plt.title('Hierarchical ClusteringAlgorithm Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')

        dendrogram(Z)

        plt.show()
        # Specify the number of desired clusters
        num_clusters = len(clusters)  # Adjust as per your requirement

        # Cut the hierarchical tree to obtain clusters
        clusters = cut_tree(Z, n_clusters=num_clusters).flatten()
        print(clusters)
        # Output the clusters and the data points within each cluster
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_points = vectors[cluster_indices]
            print(f"Cluster {cluster}:")
            print(cluster_indices)


        def extract_hierarchy(Z):
            hierarchy = {}
            for i, linkage in enumerate(Z):
                cluster = {}
                cluster["id"] = i
                cluster["left"] = int(linkage[0])
                cluster["right"] = int(linkage[1])
                cluster["distance"] = linkage[2]
                cluster["size"] = linkage[3]
                hierarchy[i] = cluster
            return hierarchy


        # Output hierarchy
        """hierarchy = extract_hierarchy(Z)
        for cluster in hierarchy.values():
            print(cluster)"""