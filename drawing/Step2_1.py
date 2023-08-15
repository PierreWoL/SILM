import io
import os
import random
import sys
import pickle

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

box_colorsM = ['#1E90FF', '#9ACD32', '#FFD700', '#EE82EE', '#FF7F50', '#7B68EE']
target_path = os.path.abspath(os.path.dirname(os.getcwd())) + "/result/Valerie/Column/TabFact/_gt_cluster.pickle"
F_cluster = open(target_path, 'rb')
KEYS = pickle.load(F_cluster)
RI = {'Agglomerative': {}}  # 'BIRCH': {},
ARI = {'Agglomerative': {}}
Purity = {'Agglomerative': {}}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "result/starmie/TabFact/All")
step1_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "result/starmie/TabFact/Subject_Col")
EMBEDMETHODS = ['SBERT_none', 'RoBERTa_none', 'SBERT_subjectheader', 'RoBERTa_subjectheader', 'SBERT_header',
                'RoBERTa_header']
"""EMBEDMETHODS = ['SBERT_Instance', 'RoBERTa_Instance',
                'SBERT_SubAttr', 'RoBERTa_SubAttr',
                'SBERT_AttrNameI', 'RoBERTa_AttrNameI']"""


def naming(filename):
    methodName = ''
    if filename.endswith('_none'):
        methodName = EMBEDMETHODS[0] if 'sbert' in filename else EMBEDMETHODS[1]
    elif filename.endswith('_subjectheader'):
        methodName = EMBEDMETHODS[2] if 'sbert' in filename else EMBEDMETHODS[3]
    elif filename.endswith('_header'):
        methodName = EMBEDMETHODS[4] if 'sbert' in filename else EMBEDMETHODS[5]
    print(filename, methodName)
    return methodName


def dataframes(folds):
    for folder in folds:
        # print(method_name,folder)
        method_name = naming(folder)
        RI['Agglomerative'][method_name] = {}
        ARI['Agglomerative'][method_name] = {}
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
            RI['Agglomerative'][method_name][name_stat] = statist_col.loc['random score', 'Agglomerative']
            # ARI['BIRCH'][method_name][name_stat] = statist_col.loc['ARI', 'BIRCH']
            ARI['Agglomerative'][method_name][name_stat] = statist_col.loc['ARI', 'Agglomerative']
            Purity['Agglomerative'][method_name][name_stat] = statist_col.loc['purity', 'Agglomerative']


def to_xlsx(df1=None, df2=None, file_path='', n1='BIRCH', n2='Agglomerative'):
    with pd.ExcelWriter(file_path) as writer:
        if df1 is not None:
            df1.to_excel(writer, sheet_name=n1, index=True)
        if df2 is not None:
            df2.to_excel(writer, sheet_name=n2, index=True)


metric = ['Rand Index', 'ARI', 'Purity']
algo = ['Agglomerative', 'BIRCH']  #


def random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# rgb_colors = [plt.cm.colors.to_rgba(color) for color in box_colors]

def box_plot(table, box_colors, y_label, name, fig_name, save=True):
    plt.figure(figsize=(10, 8))
    bp = table.boxplot(patch_artist=True, notch=True, vert=True, widths=0.5)
    # bp = plt.boxplot(table.values, labels=table.columns, patch_artist=True,notch=True,showfliers=False)
    boxes = bp.findobj(matplotlib.patches.PathPatch)
    for box, color in zip(boxes, box_colors):
        box.set(facecolor=color)
    plt.xticks(rotation=20, fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel('Embedding Methods', fontsize=10)
    plt.title(name, fontsize=14)
    plt.ylabel(y_label, fontsize=10)
    plt.subplots_adjust(top=0.9, bottom=0.18)
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
                      for label, color in zip(table.columns, box_colors)]
    # plt.legend(handles=legend_patches, loc='upper right')
    """medians = [median.get_ydata()[0] for median in bp['medians']]
    means = [mean.get_ydata()[0] for mean in bp['means']]
    xtick_labels = table.columns
    for i, median, mean in zip(range(len(xtick_labels)), medians, means):
        plt.text(i + 1 - 0.25, median, f'Median: {median:.3f}', ha='right', va='center', color='red', fontsize=8)
        plt.text(i + 1 - 0.25, mean, f'Mean: {mean:.3f}', ha='right', va='center', color='blue', fontsize=8)"""
    if save is True:
        plt.savefig(fig_name)


def naming2(metric_index, algo_index, tar_path):
    name = f"{metric[metric_index]} of Column clustering using {algo[algo_index]} clustering"
    fn = os.path.join(tar_path, f"{metric[metric_index]}_{algo[algo_index]}.png")
    y_name = metric[metric_index]
    return y_name, name, fn


def drawing():
    folds = [fn for fn in os.listdir(data_path) if "." not in fn]
    dataframes(folds)
    # df_RI_BIRCH = pd.DataFrame(RI['BIRCH'])
    # box_plot(df_RI_BIRCH,box_colorsM[0:6], 0, 0, data_path)
    df_RI_A = pd.DataFrame(RI['Agglomerative'])
    filep = os.path.join(data_path, "RI.xlsx")
    to_xlsx(df2=df_RI_A, file_path=filep)

    y_name, name, fn = naming2(0, 0, data_path)
    box_plot(df_RI_A, box_colorsM[0:6], y_name, name, fn)
    y_name, name, fn = naming2(1, 0, data_path)
    df_ARI_A = pd.DataFrame(ARI['Agglomerative'])
    filep = os.path.join(data_path, "ARI.xlsx")
    to_xlsx(df2=df_ARI_A, file_path=filep)
    box_plot(df_ARI_A, box_colorsM[:6], y_name, name, fn)
    y_name, name, fn = naming2(2, 0, data_path)
    df_P_A = pd.DataFrame(Purity['Agglomerative'])
    filep = os.path.join(data_path, "Purity.xlsx")
    to_xlsx(df2=df_P_A, file_path=filep)
    box_plot(df_P_A, box_colorsM[:6], y_name, name, fn)

drawing()
"""
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
to_xlsx(df_all_B, df_all_A, filep)"""
