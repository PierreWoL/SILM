import os
from experiments import run_exp
from multiprocessing.pool import ThreadPool as Pool


def try_parallel(dataPath):
    absolute_path = os.getcwd() + "/datasets/"
    Methods = [1]
    ground_Truth = absolute_path + dataPath + "/groundTruth.csv"
    experiment_Name = "K3_lshSBert"
    for method in Methods:
        samplePath = []
        if method == 1:
            samplePath = absolute_path + dataPath + "/feature.csv"
        if method == 2:
            samplePath = absolute_path + dataPath + "/SubjectColumn/"
        if method == 3:
            samplePath = absolute_path + dataPath + "/Test/"
        print(samplePath)
        # for k in [3, 100]:
        for embed_mode in range(2, 3):
            targetPath = dataPath + "/Method" + str(method) + "/"
            if method != 1:
                targetPath = targetPath + ("MODE" + str(embed_mode) + "/")
                experiment_Name += "Method" + str(method) +"_MODE" + str(embed_mode)
            print(ground_Truth, samplePath)
            run_exp(experiment_Name, ground_Truth, targetPath,
                    samplePath, k=3, method=method, embedding_mode=embed_mode)


if __name__ == "__main__":

    DATA_PATH = ['SOTAB', 'open_data', 'T2DV2','Test_corpus']#
    # TARGET_PATH = []
    #pool = Pool(processes=3)  # create a pool of 4 processes
    #results = pool.map(try_parallel, DATA_PATH)
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
