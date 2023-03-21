import os
from experiments import run_exp
from multiprocessing.pool import ThreadPool as Pool
from experimentalData import get_files


def try_parallel(dataPath):
    absolute_path = os.getcwd() + "/datasets/"
    Methods = [2,3]  # ,3
    ground_Truth = absolute_path + dataPath + "/groundTruth.csv"
    Max_K = 100
    for method in Methods:
        samplePath = []
        if method == 1:
            samplePath = absolute_path + dataPath + "/featureAll.csv"
        if method == 2:
            samplePath = absolute_path + dataPath + "/SubjectColumn/"
            Max_K = len(get_files(samplePath))
        if method == 3:
            samplePath = absolute_path + dataPath + "/Test/"
            Max_K = len(get_files(samplePath))
        print(samplePath)
        for k in [100]:  # 3,5, 50, 100 , -20
            for embed_mode in [2]:  #
                targetPath = dataPath + "/Method" + str(method) + "/"
                if method != 1:
                    targetPath = targetPath + ("MODE" + str(embed_mode) + "/")

                print(ground_Truth, samplePath)
                for threshold in [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    experiment_Name = "K" + str(k) + "Method" + str(method) + "_MODE" + str(
                       embed_mode)
                    run_exp(experiment_Name, ground_Truth, targetPath,
                        samplePath, threshold, k=k, method=method, embedding_mode=embed_mode)


if __name__ == "__main__":

    DATA_PATH = ['open_data', 'Test_corpus']  #,'T2DV2', 'SOTAB'
    #
    # TARGET_PATH = []
    # pool = Pool(processes=3)  # create a pool of 4 processes
    # results = pool.map(try_parallel, DATA_PATH)
    for datapath in DATA_PATH:
        try_parallel(datapath)
    """
        for i in range(0, len(DATA_PATH)):
        groundTruth = absolute_path + DATA_PATH[i] + "/groundTruth.csv"
        experimentName = "K3"
        for method in Methods:
            sample_path = []
            if method == 1:
                sample_path = [absolute_path + data_path + "/feature.csv" for data_path in DATA_PATH]
            if method == 2:
                sample_path = [absolute_path + data_path + "/SubjectColumn/" for data_path in DATA_PATH]
            if method == 3:
                sample_path = [absolute_path + data_path + "/Test/" for data_path in DATA_PATH]
            print(sample_path)
            target_path = DATA_PATH[i] + "/Method" + str(method) + "/"
            # for k in [3, 100]:
            for embed in range(1, 3):
                if method != 1:
                    target_path += ("MODE" + str(embed) + "/")
                print(groundTruth, sample_path[i])
                run_exp(experimentName, groundTruth, target_path,
                        sample_path[i], k=3, method=method, embedding_mode=embed)
    """
