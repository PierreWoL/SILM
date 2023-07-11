from Method1 import SBERT_T
import os
DATA_PATH = ['WDC',"TabFact"] #'T2DV2', 'SOTAB','open_data',



for dataset in DATA_PATH:
    target_path = 'result/embedding/starmie/vectors/%s' %  dataset
    filename =   "SBERT_.pickle"
    store_path = os.path.join(os.getcwd(), target_path, filename)
    samplePath = os.getcwd() + "/datasets/WDC/Test/"
    SBERT_T(samplePath, os.getcwd() + "/datasets/WDC/featureAll.pickle")