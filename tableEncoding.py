from Method1 import SBERT_T
import os
DATA_PATH = ['WDC'] #'T2DV2', 'SOTAB','open_data',,"TabFact"



for dataset in DATA_PATH:
    target_path = 'result/embedding/starmie/vectors/%s' %  dataset
    filename =   "SBERT_.pickle"
    store_path = os.path.join(os.getcwd(), target_path, filename)
    samplePath = "datasets/%s/Test/" %  dataset
    SBERT_T(samplePath, store_path)