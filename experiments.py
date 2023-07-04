import ast
import os
import numpy
import pandas as pd
import clustering
from Utils import mkdir


def experiment(Z, T, data_path, ground_truth, folderName, filename):
    clustering_method = ["BIRCH", "Agglomerative"]  # "DBSCAN", "GMM", "KMeans", "OPTICS",
    methods_metrics = {}
    for method in clustering_method:
        print(method)
        metric_value_df = pd.DataFrame(columns=["MI", "NMI", "AMI", "random score", "ARI", "FMI", "purity"])
        for i in range(0, 2):
            mkdir(os.getcwd() + "/result/" + method + "/" + folderName + "/")
            metric_dict = clustering.clustering_results(Z, T, data_path, ground_truth, method,
                                                        os.getcwd() + "/result/" + method + "/" + folderName + "/",
                                                        filename + str(i))[1]
            metric_df = pd.DataFrame([metric_dict])
            metric_value_df = pd.concat([metric_value_df, metric_df])
        mean_metric = metric_value_df.mean()
        methods_metrics[method] = mean_metric
    return methods_metrics




# samplePath3 = os.getcwd() + "/T2DV2/test/"
# samplePath2 = os.getcwd() + "/result/t2dv2_Subject_Columns/"
# T2DV2GroundTruth = os.getcwd() + "/T2DV2/classes_GS.csv"
# Z3, T3 = clustering.inputData(samplePath3)
# Z2, T2 = clustering.inputData(samplePath2)
# input_data, tables, exceptions = m1.SBERT(samplePath2)
# e1 = experiment(input_data, tables, samplePath2, T2DV2GroundTruth)
# e1 = experiment(input_data, tables, samplePath2, T2DV2GroundTruth)

# e2 = experiment(Z2, T2, samplePath2, T2DV2GroundTruth,'Subject_Column','e2')


def run_exp(experiment_name, GroundTruth, targetFolder, samplePath, threshold, k, method=0, embedding_mode=2):
    # samplepath 既可以是路径也可以是feature的文件绝对路径
    Z3 = pd.DataFrame()
    T3 = []
    if method == 1:
        features = pd.read_csv(samplePath)
        print(features)
        T3 = features.iloc[:, 0]
        Z3 = numpy.array([ast.literal_eval(x) for x in features.iloc[:, 1]])
        print(Z3,T3)

        """
        D = np.ones((len(T3), len(T3)))
        for i in range(len(T3)):
            D[i, i] = 0
        lsh = MinHashLSH(threshold=0.6, num_perm=256)
        minhash_dict = {}
        mg = WeightedMinHashGenerator(len(ast.literal_eval(features.iloc[:, 1][0])), 256)
        for i in range(0, len(T3)):
            m = mg.minhash(Z3[i])
            minhash_dict[T3[i]] = m
            lsh.insert(T3[i], m)
        T3_list = list(T3)
        for t in T3:
            result = lsh.query(minhash_dict[t])
            for name in result:  # index
                if name in T3 and t != name:
                    D[T3_list.index(t), T3_list.index(name)] = minhash_dict[name].jaccard(minhash_dict[t])
                    D[T3_list.index(name), T3_list.index(t)] = minhash_dict[name].jaccard(minhash_dict[t])
        Z3 = D
        """

    else:
        Z3, T3 = clustering.inputData(samplePath, threshold, k, embedding_mode=embedding_mode)
    e = experiment(Z3, T3, samplePath, GroundTruth, targetFolder, experiment_name)
    e_df = pd.DataFrame()
    for i, v in e.items():
        print(v.rename(i))
        e_df = pd.concat([e_df, v.rename(i)], axis=1)
    print(e_df)
    store_path = os.getcwd() + "/result/metrics/" + targetFolder
    mkdir(store_path)
    e_df.to_csv(store_path + experiment_name + 'DistributionLSH_metrics.csv', encoding='utf-8')


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
