import io
import os
import random
import sys
import pickle

import pandas as pd
from matplotlib import pyplot as plt

target_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/result/Valerie/Column/TabFact/_gt_cluster.pickle"
F_cluster = open(target_path, 'rb')
KEYS = pickle.load(F_cluster)
RI = {'BIRCH': {}, 'Agglomerative': {}}
ARI = {'BIRCH': {}, 'Agglomerative': {}}
Purity = {'BIRCH': {}, 'Agglomerative': {}}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
data_path = "/mnt/c/Users/1124a/Desktop/All"
step1_path = "/mnt/c/Users/1124a/Desktop/Subject_Col"
EMBEDMETHODS = ['sbert_none', 'roberta_none', 'sbert_subjectheader', 'roberta_subjectheader', 'sbert_header',
                'roberta_header']


def naming(filename):
    method_name = ''
    if filename.endswith('_none'):
        method_name = EMBEDMETHODS[0] if 'sbert' in filename else EMBEDMETHODS[1]
    if filename.endswith('_subjectheader'):
        method_name = EMBEDMETHODS[2] if 'sbert' in filename else EMBEDMETHODS[3]
    if filename.endswith('_header'):
        method_name = EMBEDMETHODS[4] if 'sbert' in filename else EMBEDMETHODS[5]
    print(filename, method_name)
    return method_name


folds = [fn for fn in os.listdir(data_path) if "." not in fn]

for folder in folds:
    # print(method_name,folder)
    method_name = naming(folder)
    RI['BIRCH'][method_name] = {}
    RI['Agglomerative'][method_name] = {}
    ARI['BIRCH'][method_name] = {}
    ARI['Agglomerative'][method_name] = {}
    Purity['BIRCH'][method_name] = {}
    Purity['Agglomerative'][method_name] = {}

    tar_folder = os.path.join(data_path, folder, "column") \
        if "column" in os.listdir(os.path.join(data_path, folder)) else os.path.join(data_path, folder)
    stat_files = [fn for fn in os.listdir(tar_folder) if fn.endswith(".csv")]

    for stat_file in stat_files:
        statist_col = pd.read_csv(os.path.join(tar_folder, stat_file), index_col=0)
        if tar_folder.endswith("column"):
            name_stat = KEYS[int(stat_file.split("_")[0])]

        else:
            name_stat = stat_file[0:-4].split("_")[0]

        RI['BIRCH'][method_name][name_stat] = statist_col.loc['random score', 'BIRCH']
        RI['Agglomerative'][method_name][name_stat] = statist_col.loc['random score', 'Agglomerative']
        ARI['BIRCH'][method_name][name_stat] = statist_col.loc['ARI', 'BIRCH']
        ARI['Agglomerative'][method_name][name_stat] = statist_col.loc['ARI', 'Agglomerative']
        Purity['BIRCH'][method_name][name_stat] = statist_col.loc['purity', 'BIRCH']
        Purity['Agglomerative'][method_name][name_stat] = statist_col.loc['purity', 'Agglomerative']


def to_xlsx(df1, df2, file_path, n1='BIRCH', n2='Agglomerative'):
    with pd.ExcelWriter(file_path) as writer:
        df1.to_excel(writer, sheet_name=n1, index=True)
        df2.to_excel(writer, sheet_name=n2, index=True)


metric = ['Rand Index', 'ARI', 'Purity']
algo = ['BIRCH','Agglomerative']

def random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
def box_plot(table, metric_index, algo_index, tar_path):
    #box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightcyan', 'lightpink']

    #rgb_colors = [plt.cm.colors.to_rgba(color) for color in box_colors]

    plt.figure(figsize=(10, 8))
    bp = table.boxplot(patch_artist=True, notch=True, vert=True, widths=0.5, showfliers=False)

    name = f"{metric[metric_index]} of Column clustering using {algo[algo_index]}"
    plt.xticks(rotation=30)
    plt.xlabel('Embedding Methods')
    plt.title(name)
    plt.ylabel(metric[metric_index])
    fn= os.path.join(tar_path,f"{metric[metric_index]}_{algo[algo_index]}.png")
    plt.savefig(fn)
    plt.show()


df_RI_BIRCH = pd.DataFrame(RI['BIRCH'])
box_plot(df_RI_BIRCH,0,0,data_path)
df_RI_A = pd.DataFrame(RI['Agglomerative'])
box_plot(df_RI_BIRCH,0,1,data_path)
filep = os.path.join(data_path, "RI.xlsx")
to_xlsx(df_RI_BIRCH, df_RI_A, filep)


df_ARI = pd.DataFrame(ARI['BIRCH'])
box_plot(df_RI_BIRCH,1,0,data_path)
df_ARI_A = pd.DataFrame(ARI['Agglomerative'])
box_plot(df_RI_BIRCH,1,1,data_path)
filep = os.path.join(data_path, "ARI.xlsx")
to_xlsx(df_ARI, df_ARI_A, filep)

df_P = pd.DataFrame(Purity['BIRCH'])
box_plot(df_RI_BIRCH,2,0,data_path)
df_P_A = pd.DataFrame(Purity['Agglomerative'])
box_plot(df_RI_BIRCH,2,1,data_path)
filep = os.path.join(data_path, "Purity.xlsx")
to_xlsx(df_P, df_P_A, filep)

files_csv = [fn for fn in os.listdir(step1_path) if fn.endswith(".csv")]
score = {'BIRCH': {}, 'Agglomerative': {}}
for file in files_csv:
    method_name = naming(file[0:-12])
    score['BIRCH'][method_name] = {}
    score['Agglomerative'][method_name] = {}
    statist_col = pd.read_csv(os.path.join(step1_path, file), index_col=0)

    score['BIRCH'][method_name]['RI'] = statist_col.loc['random score', 'BIRCH']
    score['Agglomerative'][method_name]['RI'] = statist_col.loc['random score', 'Agglomerative']
    score['BIRCH'][method_name]['ARI'] = statist_col.loc['ARI', 'BIRCH']
    score['Agglomerative'][method_name]['ARI'] = statist_col.loc['ARI', 'Agglomerative']
    score['BIRCH'][method_name]['purity'] = statist_col.loc['purity', 'BIRCH']
    score['Agglomerative'][method_name]['purity'] = statist_col.loc['purity', 'Agglomerative']
df_all_B = pd.DataFrame(score['BIRCH'])
df_all_A = pd.DataFrame(score['Agglomerative'])
filep = os.path.join(data_path, "SubColAll.xlsx")
to_xlsx(df_all_B, df_all_A, filep)
