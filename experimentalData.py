import csv
import math
import pandas as pd
import shutil
import os
from string import digits
import random


def create_experiment_directory(gt_filename, data_path, Concepts, output_path):
    """
    # Given the name of the file that contains the Ground Truth that
    # associates web table filenames with manually annotated concepts,
    # the path to the directory containing those files, and a list
    # of concepts of interest, populate the directory with the given
    # output_path with the files associated with the given Concepts.
    Parameters
    ----------
    gt_filename
    data_path
    Concepts
    output_path

    Returns
    -------

    """
    df = pd.read_csv(gt_filename, header=None)
    identifier = 0  # Name the files after their concepts
    for ind in df.index:
        if df[1][ind] in Concepts:
            src = data_path + df[0][ind].strip('.tar.gz') + '.csv'

            dst = output_path + df[1][ind] + str(identifier) + '.csv'
            identifier = identifier + 1

            print(src, dst)
            shutil.copyfile(src, dst)
    return df


def get_files(data_path):
    T=[]
    if data_path.endswith('.csv'):
        features = pd.read_csv(data_path)
        T = features.iloc[:, 0]
    else:
        T = os.listdir(data_path)
        T = [t[:-4] for t in T if t.endswith('.csv')]
        T.sort()

    return (T)


def create_gt_file(Concepts, output_path):
    T = get_files(output_path)
    gt_string = ""
    pos = 0
    for t in T:
        name = t.strip('.csv)')
        name = name.rstrip(digits)
        cluster = Concepts.index(name)
        if pos > 0:
            gt_string = gt_string + ","
        gt_string = gt_string + str(cluster)
        pos = pos + 1

    text_file = open(output_path + "cluster_gt.txt", "w")
    n = text_file.write(gt_string)
    text_file.close()


'''
Used in WDC, for obtaining the label of each table
'''


def get_concept(filepath):
    T = os.listdir(filepath)
    concept = []
    for filename in T:
        if not filename.endswith("lsh"):
            perConcept = filename.split("_")[0].lower()
            if perConcept not in concept:
                concept.append(perConcept)
    concept.sort()
    return concept

from collections import Counter
def get_concept_files(files, GroundTruth, Nochange = False):
    """
    obtain the files' classes by
    mapping its name to the ground truth
    Parameters
    ----------
    files: the test data xxxx.csv
    we have the ground truth csv that has two columns
    "filename: xxxx label: ABC"
    GroundTruth: csv file stores the ground truth
    Returns
    -------

    """

    test_gt_dic = {}
    test_gt = {}
    test_duplicate=[]
    i =0
    for file in files:
        name_without_extension = file
        if GroundTruth.get(name_without_extension) is not None:
            ground_truth = GroundTruth.get(name_without_extension)
            test_gt_dic[name_without_extension] = ground_truth
            test_duplicate.append(len(test_gt_dic.keys()))
            if type(ground_truth) is list:
                if Nochange is False:
                    for item in ground_truth:
                        if test_gt.get(item) is None:
                            test_gt[item] = []
                        test_gt[item].append(file)
                else:
                    if test_gt.get(str(ground_truth)) is None:
                        test_gt[str(ground_truth)] = []
                    test_gt[str(ground_truth)].append(file)
            else:
                if test_gt.get(ground_truth) is None:
                    test_gt[ground_truth] = []
                test_gt[ground_truth].append(file)
        i+=1
    return test_gt_dic, test_gt


def get_random_train_data(data_path, train_path, portion):
    prefiles = os.listdir(data_path)
    files = []
    for file in prefiles:
        if not file.strip(".csv").endswith("Metadata"):
            files.append(file)
    number_of_files = len(files)
    print(number_of_files)
    selected_number = math.floor(number_of_files * portion)
    samples = random.sample(files, selected_number)
    for sample in samples:
        if data_path.endswith("/"):
            shutil.copy(data_path + sample, train_path)
        else:
            shutil.copy(data_path + "/" + sample, train_path)


def create_test_data(gt_filename, data_path, output_path):
    # Given the name of the file that contains the Ground Truth that
    # associates web table filenames with manually annotated concepts,
    # the path to the directory containing those files, and a list
    # of concepts of interest, populate the directory with the given
    # output_path with the files associated with the given Concepts.
    df = pd.read_csv(gt_filename, header=None)
    if not data_path.endswith("/"):
        data_path += "/"
    if not output_path.endswith("/"):
        output_path += "/"
    for index, row in df.iterrows():
        filename = row[0].strip('.tar.gz')
        src = data_path + filename + '.csv'
        metadata = data_path + "metadata/" + filename + 'Metadata.csv'
        dst = output_path + filename + '.csv'
        metadst = output_path + "metadata/" + filename + 'Metadata.csv'
        shutil.copyfile(src, dst)
        shutil.copyfile(metadata, metadst)

"""
T2DV2Path = os.getcwd() + "/T2DV2/tables/test/"
samplePath = os.getcwd() + "/T2DV2/test/"
ground_truth = os.getcwd() + "/T2DV2/classes_GS.csv"
create_test_data(ground_truth, T2DV2Path, samplePath)
"""

"""
WDCFilePath = os.getcwd() + "/WDC/CPA_Validation/Validation/Table/"
T2DV2Path = os.getcwd() + "/T2DV2/"
samplePath = os.getcwd() + "/T2DV2/test/"
WDCsamplePath = os.getcwd() + "/WDC/CPA_Validation/Validation/Table/test/"
get_random_train_data(T2DV2Path, samplePath, 0.9)
"""
# get_random_train_data(WDCFilePath, WDCsamplePath, 0.1)

'''
Concepts = ['Animal','Bird','City','Museum','Plant','University']
df = create_experiment_directory('T2DGroundTruth/classes_complete.csv', \
                                 'T2DGroundTruth/tables_complete/', Concepts, \
                                 'T2DGroundTruth/city_things/')
create_gt_file(Concepts,'T2DGroundTruth/city_things/')
'''
