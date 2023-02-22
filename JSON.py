import csv
import json
import os
from time import sleep
from pandas import json_normalize
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
            f = open(filename, errors='ignore')
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
            if folder != ".DS_Store":
                folder_files = self.get_files(self.folder + folder + "/")
                for file in folder_files:
                    csv_folder = os.getcwd() + "/datasets/WDC_corpus/"
                    json_file = self.folder + folder + "/" + file
                    json_csv(csv_folder, json_file, file)
                    #json_csv_Test(json_file)
                    write_line(ground_truth_csv, [file, folder])


# folder = T2DV2Path = os.getcwd() + "/T2DV2/classes_GS.csv"
# folder = T2DV2Path = os.getcwd() + "/WDC1/Country/Country_africansafaris.co.nz_September2020.json"

def json_csv_Test(file):
    with open(file, 'r') as f:
        df = pd.DataFrame()
        for line in f:
            data = json.loads(line)
            temp_df = pd.DataFrame(data)
            df = pd.concat([df, temp_df])
        print(df.iloc[0:3, ])


def json_csv(folder_target, file, filename):
    with open(file, 'r') as f:
        df = pd.DataFrame()
        for line in f:
            data = json.loads(line)
            df_line = pd.Series(data)
            df = pd.concat([df, df_line], axis=1)
        # print(file,df)
        df = df.T.reset_index(drop=True)
        # column_dict, column_list, is_complicated = detect_nested(df)
        df = check_normalized(file, df)
        print(df.T)
        df.to_csv(folder_target + filename.strip(".json") + ".csv", index=False)


def detect_nested(df):
    column_dict = {}
    column_list = {}
    is_complicated = {}
    for column_index in range(0, df.shape[1]):
        if type(df.iloc[-1, column_index]) == dict:
            column_dict[column_index] = df.iloc[:, column_index].name
        if type(df.iloc[-1, column_index]) == list:
            if type(df.iloc[-1, column_index][0]) == dict:
                is_complicated[column_index] = df.iloc[:, column_index].name
            else:
                column_list[column_index] = df.iloc[:, column_index].name
    return column_dict, column_list, is_complicated


def normalize_tables(df: pd.DataFrame):
    column_dict, column_list, is_complicated = detect_nested(df)
    if column_dict:
        for index, column_name in column_dict.items():
            try:
                nor_col = json_normalize(df.iloc[:, index].tolist()).add_prefix(df.iloc[:, index].name + ".")
                df = pd.concat([df, nor_col], axis=1).reset_index(drop=True)
            except (AttributeError, TypeError) as e:
                continue

    if is_complicated:
        for index, column_name in is_complicated.items():
            try:
                nor_col = json_normalize(df.iloc[:, index].apply(list_of_dicts).tolist()).add_prefix(
                    df.iloc[:, index].name + ".")
                df = pd.concat([df, nor_col], axis=1).reset_index(drop=True)
            except (AttributeError, TypeError, IndexError) as e:
                continue

    if column_list:
        for index, column_name in column_list.items():
            try:
                values = df.iloc[:, index].apply(list_to_str)
                df[column_name] = list(values)
            except (AttributeError, TypeError) as e:
                continue
    df = df.drop(df.columns[list(column_dict.keys()) + list(is_complicated.keys()) + list(column_list.keys())], axis=1)
    return df


def check_normalized(file, df):
    column_dict, column_list, is_complicated = detect_nested(df)
    if not column_dict and not column_list and not is_complicated:
        return df
    else:
        df = normalize_tables(df)
        check_normalized(file,df)
    return df


def list_to_str(x):
    if type(x) == list:
        return ','.join(x)
    return x


def list_of_dicts(ld: dict):
    '''
    Create a mapping of the tuples formed after
    converting json strings of list to a python list
    '''
    return dict([(list(d.values())[1], list(d.values())[0]) for d in ld])


def write_line(csv_name, row: list):
    # check if file exists
    # file_exists = os.path.isfile(csv_name)
    # if file_exists is True:
    with open(csv_name, "a", newline='') as csvfile:
        # create CSV writer object
        csv_writer = csv.writer(csvfile)
        # write data row
        csv_writer.writerow(row)


wdc = WDC(os.getcwd() + "/datasets/WDC1/")
wdc.trans_csv(os.getcwd() + "/datasets/" + "WDC_corpus_ground_truth.csv")
# for key in gt_clusters:
#   shutil.copy(samplePath + key+".csv", samplePath)
# groundTruthWDCTest = ed.get_concept(ed.WDCsamplePath)
# print(groundTruthWDCTest)
# samplePath = os.getcwd() + "/T2DV2/Test/"
# indexes = create_or_find_indexes(ed.WDCsamplePath)
# parameters = dbscan_param_search(ed.WDCsamplePath, indexes)
# print(parameters)
# clusters = cluster_discovery(ed.WDCsamplePath,parameters)
