import csv
import math
import pandas as pd
import shutil
import os
from string import digits
import random


def create_experiment_directory(gt_filename, data_path, Concepts, output_path):
    # Given the name of the file that contains the Ground Truth that
    # associates web table filenames with manually annotated concepts,
    # the path to the directory containing those files, and a list
    # of concepts of interest, populate the directory with the given
    # output_path with the files associated with the given Concepts.


    df = pd.read_csv(gt_filename, header=None)
    identifier = 0 # Name the files after their concepts
    for ind in df.index:
        if (df[1][ind] in Concepts):
            src = data_path + df[0][ind].strip('.tar.gz') + '.csv'

            dst = output_path + df[1][ind] + str(identifier) + '.csv'
            identifier = identifier + 1

            print(src, dst)
            shutil.copyfile(src, dst)
    return df


def get_files(data_path):
    T = os.listdir(data_path)
    T = [t[:-4] for t in T if t.endswith('.csv')]
    T.sort()
    return(T)


def create_gt_file(Concepts,output_path):
    T = get_files(output_path)
    gt_string = ""
    pos = 0
    for t in T:
        name = t.strip('.csv)')
        name = name.rstrip(digits)
        cluster = Concepts.index(name)
        if (pos > 0):
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
            if  perConcept not in concept:
                concept.append(perConcept)
    concept.sort()
    return (concept)




def get_concept_files(files, GroundTruth):
    test_gt_dic={}
    test_gt ={}
    truth=[]
    for file in files:
        if GroundTruth.get(file.strip(".csv")) !=None:
            ground_truth = GroundTruth.get(file.strip(".csv"))
            test_gt_dic[file.strip(".csv")] = ground_truth
            truth.append(ground_truth)
            if test_gt.get(ground_truth)== None:
                test_gt[ground_truth] = []
            test_gt[ground_truth].append(file)

    return test_gt_dic,test_gt,truth



def get_random_train_data(data_path,train_path, portion):
  prefiles = os.listdir(data_path)
  files=[]
  for file in prefiles:
        if not file.strip(".csv").endswith("Metadata"):
          files.append(file)
  number_of_files = len(files)
  print(number_of_files)
  selected_number = math.floor(number_of_files*portion)
  samples = random.sample(files, selected_number)
  for sample in samples:
      if data_path.endswith("/"):
              shutil.copy(data_path + sample, train_path )
      else:
              shutil.copy(data_path+"/"+sample, train_path)




WDCFilePath = "C:/Users/1124a/OneDrive - The University of Manchester/CurrentDataset-main/WDC/CPA_Validation/Validation/Table/"
T2DV2Path = "C:/Users/1124a/OneDrive - The University of Manchester/CurrentDataset-main/T2DV2/"
samplePath = "C:/Users/1124a/OneDrive - The University of Manchester/CurrentDataset-main/T2DV2/Test/"
WDCsamplePath = "C:/Users/1124a/OneDrive - The University of Manchester/CurrentDataset-main/WDC/CPA_Validation/Validation/Table/Test/"
#get_random_train_data(T2DV2Path, samplePath, 0.9)
#get_random_train_data(WDCFilePath, WDCsamplePath, 0.1)

'''
Concepts = ['Animal','Bird','City','Museum','Plant','University']
df = create_experiment_directory('T2DGroundTruth/classes_complete.csv', \
                                 'T2DGroundTruth/tables_complete/', Concepts, \
                                 'T2DGroundTruth/city_things/')
create_gt_file(Concepts,'T2DGroundTruth/city_things/')
'''

