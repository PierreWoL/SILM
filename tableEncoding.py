from Method1 import SBERT_T
import os
DATA_PATH = ['Test_corpus'] #'T2DV2', 'SOTAB','open_data',


samplePath = os.getcwd() + "/datasets/WDC/Test/"
SBERT_T(samplePath, os.getcwd() + "/datasets/WDC/featureAll.pickle")