import os

import pandas as pd

from CA_evaluation import column_gts

gt_clusters, ground_t, gt_cluster_dict = column_gts("WDC", superclass=True)
print(gt_clusters, "\n", "\n",ground_t,  "\n","\n", gt_cluster_dict)
def findTables(dataset):
	reference_path = f"datasets/{dataset}/"
	gt_csv = pd.read_csv(os.path.join(reference_path, "groundTruth.csv"))
	class_to_files = gt_csv.groupby('superclass')['fileName'].apply(set).to_dict()
	return class_to_files
class_file_Dict = findTables("WDC")
print(class_file_Dict)
