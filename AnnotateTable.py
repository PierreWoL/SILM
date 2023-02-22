import os
import csv
import pandas as pd


class SOTAB:
    def __init__(self, datafolder):
        self.folder = datafolder

    def get_files(self, csv_folder):
        T = os.listdir(csv_folder)
        T = [t for t in T if t.endswith('.csv')]
        T.sort()
        return T

    def trans_csv(self, ground_truth_csv):
        folders = os.listdir(self.folder)
        print(folders)
        for folder in folders:
            if folder != ".DS_Store" and folder != "gt_openData.csv":
                folder_files = self.get_files(self.folder + folder + "/")
                for file in folder_files:
                    json_file = self.folder + folder + "/" + file
                    write_line(ground_truth_csv, [file, folder])


def write_line(csv_name, row: list):
    # check if file exists
    # file_exists = os.path.isfile(csv_name)
    # if file_exists is True:
    with open(csv_name, "a", newline='') as csvfile:
        # create CSV writer object
        csv_writer = csv.writer(csvfile)
        # write data row
        csv_writer.writerow(row)


def annotate_sotab(folder, gt_csv_file):
    if not folder.endswith("/"):
        folder = folder + "/"
    T = os.listdir(folder)
    T = [t for t in T if t.endswith('.csv')]
    T.sort()
    gt_csv = pd.read_csv(open(gt_csv_file, errors='ignore'))
    for file in T:
        """
        f = open(folder + file, errors='ignore')
        table_gt = gt_csv[gt_csv['table_name'] == file.strip(".csv") + ".json.gz"]
        table = pd.read_csv(f)
        print(table.columns)
        for index, row in table_gt.iterrows():
            table = table.rename(columns={str(row["column_index"]): row["label"]})
        print(table)
        table.to_csv(folder + file,index=False)
        """
        label = file.split("_")[0]
        write_line(os.getcwd() + "/datasets/SOTAB/ground_truth.csv", [file, label])


annotate_sotab(os.getcwd() + "/datasets/SOTAB/Table/", os.getcwd() + "/datasets/SOTAB/CPA_validation_gt.csv")

# sotab = SOTAB(os.getcwd() + "/datasets/open_data/")
# sotab.trans_csv(os.getcwd() + "/datasets/open_data/" + "gt_openData.csv")
