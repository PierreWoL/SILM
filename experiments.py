import ast
import os
import numpy
import pandas as pd
import clustering


def experiment(Z, T, data_path, ground_truth, folderName, filename):
    clustering_method = [ "BIRCH", "Agglomerative"]  #"DBSCAN", "GMM", "KMeans", "OPTICS",
    methods_metrics = {}
    for method in clustering_method:
        print(method)
        metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
        for i in range(0, 4):
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


def run_exp(experiment_name, GroundTruth, targetFolder, samplePath, k, method=0):
    # samplepath 既可以是路径也可以是feature的文件绝对路径
    Z3 = pd.DataFrame()
    T3 = []
    if method == 1:
        features = pd.read_csv(samplePath)
        T3 = features.iloc[:, 0]
        # Z3 = features.iloc[:,1]
        Z3 = numpy.array([ast.literal_eval(x) for x in features.iloc[:, 1]])
        # print(Z3)
    else:
        Z3, T3 = clustering.inputData(samplePath, k)
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
# "/datasets/T2DV2/test/"
# "/datasets/T2DV2/feature.csv"
# "/datasets/T2DV2/classes_GS.csv"
# "/datasets/Test_corpus/Test/"
# "/datasets/WDC_corpus_ground_truth.csv"
# "/datasets/SOTAB/Table/Test/"
# "/datasets/SOTAB/feature.csv"
#  "/datasets/SOTAB/ground_truth.csv"
# "/datasets/open_data/SubjectColumn/"
#  "/datasets/open_data/gt_openData.csv"
# "/datasets/T2DV2/SubjectColumn/"
# "/datasets/T2DV2/classes_GS.csv"

samplePath = os.getcwd() + "/datasets/Test_corpus/feature.csv"
experimentName = "Test_corpus_M1"
groundTruth = os.getcwd() + "/datasets/WDC_corpus_ground_truth.csv"
targetPath = "Test_corpus/M1"
klist = [3, 5]

for k in klist:
    run_exp(experimentName + "_" + str(k) + "k", groundTruth, targetPath + "_" + str(k) + "k/", samplePath, k, method=1)
