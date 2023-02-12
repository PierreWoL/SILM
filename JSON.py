import json
import os
import pandas as pd
import numpy as np



newFolder =os.getcwd() + "/T2DV2/tables/test/"
#os.makedirs(newFolder, exist_ok=True)
newFolder_meta =os.getcwd() + "/T2DV2/tables/test/metadata/"
#os.makedirs(newFolder_meta, exist_ok=True)



def loadJsonFile(folder):
    files = os.listdir(folder)
    for filename in files:
        filename = folder + filename
        f = open(filename, )
        file = json.loads(f)
        print(file)


folder = T2DV2Path = os.getcwd() + "/T2DV2/tables/"

files = os.listdir(folder)

for absoluteFilename in files:
    if absoluteFilename.endswith(".json"):
        filename = folder + absoluteFilename
        file = open(filename, encoding='utf-8', errors='ignore')
        print(filename)
        data = json.load(file)
        arrayTable = np.transpose(data["relation"])
        Table = pd.DataFrame(arrayTable[1:], columns=arrayTable[0])
        print(Table)
        Table.to_csv(newFolder + absoluteFilename.split(".")[0] + ".csv", index=False)
        data.pop("relation")
        Metadata = pd.DataFrame(data.items())
        Metadata.to_csv(newFolder_meta + absoluteFilename.split(".")[0] + "Metadata" + ".csv", index=False,
                        header=False)
'''
folder = T2DV2Path = os.getcwd() + "/T2DV2/classes_GS.csv"


'''
'''
file = open("D:/CurrentDataset-main/extended_instance_goldstandard/tables/1146722_1_7558140036342906956.json",encoding='utf-8', errors='ignore')
data = json.load(file)
arrayTable = np.transpose(data["relation"])
Table = pd.DataFrame(arrayTable[1:], columns=arrayTable[0])
print(Table)
Table.to_csv(newFolder+"1146722_1_7558140036342906956.csv", index=False)
data.pop("relation")
Metadata = pd.DataFrame(data.items())
Metadata.to_csv(newFolder+"1146722_1_7558140036342906956Metadata.csv",index=False,header=False)
print(Metadata)
'''
