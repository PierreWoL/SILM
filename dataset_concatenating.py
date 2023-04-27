import os
import shutil

dest_path = os.getcwd()+"/datasets/WDC/"
path = os.getcwd()+"/datasets/"
datasets = ["SOTAB","T2DV2","Test_corpus"]
for dataset in datasets:
    for i, filename in enumerate(os.listdir(os.path.join(path,dataset))):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Construct the full path of the source and destination files
            source_file = os.path.join(path,dataset, filename)
            destination_file = os.path.join(dest_path, f"{os.path.basename(dataset)}_{i}.csv")
            # Use shutil.copy() to copy the file to the destination folder
            shutil.copy(source_file, destination_file)