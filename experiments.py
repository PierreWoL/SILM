import os
import pandas as pd
import clustering
import Method1 as m1


def experiment(Z, T, data_path, ground_truth, folderName,filename):
    clustering_method = [ "GMM", "DBSCAN","KMeans","DBSCAN","OPTICS",  "BIRCH", "Agglomerative"]#
    methods_metrics = {}
    for method in clustering_method:
        print(method)
        metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
        for i in range(0, 2):
            metric_dict = clustering.clustering_results(Z, T, data_path, ground_truth, method,
                                                        os.getcwd()+"/result/"+method+"/"+folderName+"/",filename+str(i))
            metric_df = pd.DataFrame([metric_dict])
            metric_value_df = pd.concat([metric_value_df, metric_df])
        mean_metric = metric_value_df.mean()
        methods_metrics[method] = mean_metric
    return methods_metrics


samplePath3 = os.getcwd() + "/T2DV2/test/"
samplePath2 = os.getcwd() + "/result/t2dv2_Subject_Columns/"
T2DV2GroundTruth = os.getcwd() + "/T2DV2/classes_GS.csv"
Z3, T3 = clustering.inputData(samplePath3)
Z2, T2 = clustering.inputData(samplePath2)
# input_data, tables, exceptions = m1.SBERT(samplePath2)
# e1 = experiment(input_data, tables, samplePath2, T2DV2GroundTruth)

#e2 = experiment(Z2, T2, samplePath2, T2DV2GroundTruth,'Subject_Column','e2')
e3 = experiment(Z3, T3, samplePath3, T2DV2GroundTruth,'baseline','e3')
e3_df = pd.DataFrame()
for i,v in e3.items():
    print(v.rename(i))
    e3_df = pd.concat([e3_df, v.rename(i)],axis=1)
print(e3_df)
e3_df.to_csv('e3_metrics.csv', encoding='utf-8')


