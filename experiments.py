import os
import pandas as pd
import clustering
import Method1 as m1
#run_exp(experimentName, groundTruth, targetpath, samplePath_SOTAB)
# e = experiment(Z3, T3, samplePath, GroundTruth, targetFolder, experiment_name)
def experiment(Z, T, data_path, ground_truth, folderName, filename):
    clustering_method = ["DBSCAN","GMM", "KMeans",  "OPTICS", "BIRCH", "Agglomerative"]  #
    methods_metrics = {}
    for method in clustering_method:
        print(method)
        metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
        for i in range(0, 3):
            mkdir(os.getcwd() + "/result/" + method + "/" + folderName + "/")
            metric_dict = clustering.clustering_results(Z, T, data_path, ground_truth, method,
                                                        os.getcwd() + "/result/" + method + "/" + folderName + "/",
                                                        filename + str(i))
            metric_df = pd.DataFrame([metric_dict])
            metric_value_df = pd.concat([metric_value_df, metric_df])
        mean_metric = metric_value_df.mean()
        methods_metrics[method] = mean_metric
    return methods_metrics


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
    else:
        print("---  There is this folder!  ---")


# samplePath3 = os.getcwd() + "/T2DV2/test/"
# samplePath2 = os.getcwd() + "/result/t2dv2_Subject_Columns/"
# T2DV2GroundTruth = os.getcwd() + "/T2DV2/classes_GS.csv"
# Z3, T3 = clustering.inputData(samplePath3)
# Z2, T2 = clustering.inputData(samplePath2)
# input_data, tables, exceptions = m1.SBERT(samplePath2)
# e1 = experiment(input_data, tables, samplePath2, T2DV2GroundTruth)
# e1 = experiment(input_data, tables, samplePath2, T2DV2GroundTruth)

# e2 = experiment(Z2, T2, samplePath2, T2DV2GroundTruth,'Subject_Column','e2')


def run_exp(experiment_name, GroundTruth, targetFolder, samplePath,k):
    Z3, T3 = clustering.inputData(samplePath,k)
    e = experiment(Z3, T3, samplePath, GroundTruth, targetFolder, experiment_name)
    e_df = pd.DataFrame()
    for i, v in e.items():
        print(v.rename(i))
        e_df = pd.concat([e_df, v.rename(i)], axis=1)
    print(e_df)
    e_df.to_csv(os.getcwd() + "/result/metrics/" + experiment_name + '_metrics.csv', encoding='utf-8')



"""
T2DV2GroundTruth = os.getcwd() + "/T2DV2/classes_GS.csv"
samplePath_SOTAB = os.getcwd() + "/datasets/SOTAB/Table/"
experimentName = "sotab_eall"
groundTruth = os.getcwd() + "/datasets/SOTAB/ground_truth.csv"
targetpath = "SOTAB_baseline/"


samplePath_SOTAB = os.getcwd() + "/datasets/open_data/test/"
experimentName = "opendata"
groundTruth = os.getcwd() + "/datasets/open_data/gt_openData.csv"
targetpath = "openData_baseline/"
"""


samplePath = os.getcwd() + "/datasets/Test_corpus/SubjectColumn/"
experimentName = "WDC_corpus_SubjectColumna"
groundTruth = os.getcwd() + "/datasets/WDC_corpus_ground_truth.csv"
targetPath = "Test_corpus/SubjectColumna"
klist = [25,30,40,50]
for k in klist:
    run_exp(experimentName+"_"+str(k)+"k", groundTruth, targetPath+"_"+str(k)+"k/", samplePath,k)
