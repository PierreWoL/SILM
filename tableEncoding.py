from Method1 import SBERT_T
import os
DATA_PATH = ['Test_corpus'] #'T2DV2', 'SOTAB','open_data',

for path in DATA_PATH:
    samplePath = os.getcwd() + "/datasets/" + path + "/Test/"
    SBERT_T(samplePath, os.getcwd() + "/datasets/" + path + "/"+ "featureAll.csv")