import os
import shutil
import csv
import pandas as pd

dest_path = os.getcwd() + "/datasets/WDC/"
path = os.getcwd() + "/datasets/"
datasets = ["SOTAB", "T2DV2", "Test_corpus"]
for dataset in datasets:
    ground_truth = pd.read_csv(os.path.join(path, dataset, "groundTruth.csv"))
    feature = pd.read_csv(os.path.join(path, dataset, "feature.csv"))
    feature_all = pd.read_csv(os.path.join(path, dataset, "featureAll.csv"))

    for i, filename in enumerate(os.listdir(os.path.join(path, dataset, "Test"))):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file = filename[:-4]
            file_label = ground_truth[ground_truth.iloc[:, 0].str.contains(file)].iloc[0, 1]
            Label_path = os.path.join(dest_path, "groundTruth.csv")

            source_file = os.path.join(path, dataset, "Test", filename)
            destination_file = os.path.join(dest_path + "Test/", f"{os.path.basename(dataset)}_{i}.csv")
            with open(Label_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([dataset + "_" + str(i), file_label])
            #print(file)
            file_feature = feature[feature.iloc[:, 0].str.contains(file)]
            file_feature_all = feature_all[feature_all.iloc[:, 0].str.contains(file)]
            #print(file_feature,file_feature_all)
            if file_feature.shape[0] > 0:
                feature_path = os.path.join(dest_path, "feature.csv")
                with open(feature_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([dataset + "_" + str(i), file_feature.iloc[0,1]])
            else:
                print("No feature exists! " ,dataset + "_" + str(i),filename)
            if file_feature_all.shape[0] > 0:
                feature_all_path = os.path.join(dest_path, "featureAll.csv")
                with open(feature_all_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([dataset + "_" + str(i), file_feature_all.iloc[0,1]])
            else:
                print("No feature all exists! ", filename)
            # Construct the full path of the source and destination files
            # Use shutil.copy() to copy the file to the destination folder
            shutil.copy(source_file, destination_file)
