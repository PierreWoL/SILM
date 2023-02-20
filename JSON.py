import csv
import json
import os
import pandas as pd
import numpy as np


# newFolder = os.getcwd() + "/T2DV2/tables/test/"
# os.makedirs(newFolder, exist_ok=True)
# newFolder_meta = os.getcwd() + "/T2DV2/tables/test/metadata/"


# os.makedirs(newFolder_meta, exist_ok=True)

class T2DV2:
    def __init__(self, folder):
        self.folder = folder

    def loadJsonFile(self):
        files = os.listdir(self.folder)
        for filename in files:
            filename = self.folder + filename
            f = open(filename, )
            file = json.loads(f)
            print(file)

    def copy(self, target_path):
        files = os.listdir(self.folder)
        for absoluteFilename in files:
            if absoluteFilename.endswith(".json"):
                filename = self.folder + absoluteFilename
                file = open(filename, encoding='utf-8', errors='ignore')
                print(filename)
                data = json.load(file)
                arrayTable = np.transpose(data["relation"])
                Table = pd.DataFrame(arrayTable[1:], columns=arrayTable[0])
                print(Table)
                Table.to_csv(target_path + absoluteFilename.split(".")[0] + ".csv", index=False)
                data.pop("relation")
                Metadata = pd.DataFrame(data.items())
                Metadata.to_csv(target_path + absoluteFilename.split(".")[0] + "Metadata" + ".csv", index=False,
                                header=False)


class WDC:
    def __init__(self, datafolder):
        self.folder = datafolder

    def get_files(self, json_folder):
        T = os.listdir(json_folder)
        T = [t for t in T if t.endswith('.json')]
        T.sort()
        return T

    def trans_csv(self, ground_truth_csv):
        folders = os.listdir(self.folder)
        for folder in folders:
            if folder == "CreativeWork" or folder =="Place"\
                     or folder =="SportsEvent" or folder =="Event":
                # folder !="ShoppingCenter" and folder !="City" and folder !=".DS_Store" and \
                # folder != "Person" and folder !="Product" \
                folder_files = self.get_files(self.folder + folder + "/")
                for file in folder_files[:10]:
                    csv_folder = os.getcwd() + "/WDC_corpus/"
                    json_file = self.folder + folder + "/" + file
                    print(json_file)
                    json_csv(csv_folder, json_file, file)
                    write_line(ground_truth_csv, [file, folder])




# folder = T2DV2Path = os.getcwd() + "/T2DV2/classes_GS.csv"
# folder = T2DV2Path = os.getcwd() + "/WDC1/Country/Country_africansafaris.co.nz_September2020.json"


def json_csv(folder_target, file, filename):
    with open(file, 'r') as f:
        df = pd.DataFrame()
        for line in f:
            data = json.loads(line)
            df_line = pd.Series(data)
            # print(df_line,df_line.index)
            df = pd.concat([df, df_line], axis=1)
        df = df.T
        df.to_csv(folder_target + filename.strip(".json") + ".csv", index=False)


def write_line(csv_name, row: list):
    # check if file exists
    # file_exists = os.path.isfile(csv_name)
    # if file_exists is True:
    with open(csv_name, "a", newline='') as csvfile:
        # create CSV writer object
        csv_writer = csv.writer(csvfile)
        # write data row
        csv_writer.writerow(row)


wdc = WDC(os.getcwd() + "/WDC1/")
wdc.trans_csv("ground_truth.csv")
# for key in gt_clusters:
#   shutil.copy(samplePath + key+".csv", samplePath)
# groundTruthWDCTest = ed.get_concept(ed.WDCsamplePath)
# print(groundTruthWDCTest)
# samplePath = os.getcwd() + "/T2DV2/Test/"
# indexes = create_or_find_indexes(ed.WDCsamplePath)
# parameters = dbscan_param_search(ed.WDCsamplePath, indexes)
# print(parameters)
# clusters = cluster_discovery(ed.WDCsamplePath,parameters)