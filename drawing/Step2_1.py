import io
import os
import random
import sys
import pickle
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
def get_n_colors(n):
    return sns.color_palette("husl", n)

dataset = "WDC"

target_path = os.path.abspath(os.path.dirname(os.getcwd())) + f"/result/SILM/Column/{dataset}/_gt_cluster.pickle"
F_cluster = open(target_path, 'rb')
KEYS = pickle.load(F_cluster)
print( len(KEYS),KEYS)
RI = {'Agglomerative': {} }  # 'BIRCH': {},

Purity = {'Agglomerative': {} }

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), f"result/SILM/{dataset}/All")

"""EMBEDMETHODS = ['SBERT_Instance', 'RoBERTa_Instance',
                'SBERT_SubAttr', 'RoBERTa_SubAttr',
                'SBERT_AttrNameI', 'RoBERTa_AttrNameI']"""
def reName(fileName:str):
    re_name = ""
    if fileName.startswith("cl_"):
        fileName = fileName[3:]
        fileName_split = fileName.split("_lm_")
        Aug_op = fileName_split[0]
        txt = fileName_split[1]
        split_txt =  txt.split("_")

        is_SubCol = False
        is_col = False
        if split_txt[-1] == "subCol":
            LM, metadata = split_txt[0], split_txt[-2]
            is_SubCol = True
        elif split_txt[-1] == "column":
            LM, metadata = split_txt[0], split_txt[-2]
            is_col = True
        else:
            LM, metadata = split_txt[0], split_txt[-1]
        meta = ""
        if metadata == "none":
            meta = "I"
        elif metadata == "header":
            meta = "HI"
        elif metadata == "subjectheader":
            meta = "SHI"
        re_name = Aug_op + "_" + LM + "_" + meta
        if is_SubCol is True:
            re_name = Aug_op + "_" + LM + "_" + meta+"_"+"subAttr"
        if is_col is True:
            re_name = Aug_op + "_" + LM + "_" + meta+"_"+"Column"
    elif fileName.startswith("Pretrain_"):
        re_name+="Pre_"
        fileName = fileName.split("Pretrain_")[1].split("_")
        meta = ""
        if fileName[-2] =="none":
            meta = "I"
        elif   fileName[-2] == "header":
            meta = "HI"
        elif fileName[-2] =="subjectheader":
            meta = "SHI"
        re_name +=fileName[0]+"_"+meta
    elif fileName.startswith("D3L"):
        re_name += "D3L"
    print(fileName,re_name)
    return re_name
test = "cl_sample_cells_lm_sbert_head_column_0_subjectheader_subCol_metrics.csv".split("_metrics.csv")[0]


def dataframes(folds,clustering_algo):
    for folder in folds:
        method_name = reName(folder)
        print(method_name,folder)
        RI[clustering_algo][method_name] = {}
        #ARI[clustering_algo][method_name] = {}
        Purity[clustering_algo][method_name] = {}
        tar_folder = os.path.join(data_path, folder, "column") \
            if "column" in os.listdir(os.path.join(data_path, folder)) else os.path.join(data_path, folder)
        print(tar_folder)
        stat_files = [fn for fn in os.listdir(tar_folder) if fn.endswith(".csv")]
        for stat_file in stat_files:
            print(stat_file)
            statist_col = pd.read_csv(os.path.join(tar_folder, stat_file), index_col=0)
            if tar_folder.endswith("column"):
                name_stat = KEYS[int(stat_file.split("_")[0])]
            else:
                name_stat = stat_file[0:-4].split("_")[0]
            RI[clustering_algo][method_name][name_stat] = statist_col.loc['random Index', clustering_algo]
            #ARI[clustering_algo][method_name][name_stat] = statist_col.loc['ARI', clustering_algo]
            Purity[clustering_algo][method_name][name_stat] = statist_col.loc['purity', clustering_algo]




metric = ['Rand Index', 'ARI', 'Purity']
algo = ['Agglomerative']  #'Agglomerative',


def random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# rgb_colors = [plt.cm.colors.to_rgba(color) for color in box_colors]

def box_plot(table, box_colors, y_label, name, fig_name, save=True):
    plt.figure(figsize=(35,20))
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



def to_xlsx_pair(df1=None, df2=None, file_path='', n1='BIRCH', n2='Agglomerative'):
    with pd.ExcelWriter(file_path) as writer:
        if df1 is not None:
            df1.to_excel(writer, sheet_name=n1, index=True)
        if df2 is not None:
            df2.to_excel(writer, sheet_name=n2, index=True)

def to_xlsx(df, file_path='', name='Agglomerative'):
    if not os.path.exists(file_path):
        # Create a new Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=name, index=True)
    else:
        # Append to existing Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=name, index=True)


def naming2(metric_index, algo_index, tar_path):
    name = f"{metric[metric_index]} of Column clustering using {algo[algo_index]} clustering"
    fn = os.path.join(tar_path, f"{metric[metric_index]}_{algo[algo_index]}.png")
    y_name = metric[metric_index]
    return y_name, name, fn
def drawing():

    folds = [fn for fn in os.listdir(data_path) if "." not in fn and "old" not in fn]
    folds.sort()
    colors = get_n_colors(len(folds))
    for algorithm in algo:
        dataframes(folds,algorithm)
        df_RI_A = pd.DataFrame(RI[algorithm])
        print(algorithm)
        filep = os.path.join(data_path, "RI.xlsx")
        to_xlsx(df_RI_A, file_path=filep,name = algorithm)
        y_name, name, fn = naming2(0, algo.index(algorithm), data_path)
        #box_plot(df_RI_A, colors, y_name, name, fn)



        y_name, name, fn = naming2(2,  algo.index(algorithm), data_path)
        df_P_A = pd.DataFrame(Purity[algorithm])
        filep = os.path.join(data_path, "Purity.xlsx")
        to_xlsx(df_P_A, file_path=filep,name = algorithm)
        #box_plot(df_P_A, colors, y_name, name, fn)


#drawing()
"""
df_RI_A = pd.read_excel(os.path.join(data_path, "RI.xlsx"),sheet_name=algo[0])
print(df_RI_A,len(df_RI_A.columns)-1)
colors = get_n_colors(len(df_RI_A.columns))
fn = os.path.join(data_path, f"{metric[0]}_{algo[0]}.png")
name = f"{metric[0]} of Column clustering using {algo[0]} clustering"
box_plot(df_RI_A, colors, metric[0],name , fn)"""

"""df_RI_A = pd.read_excel(os.path.join(data_path, "Purity.xlsx"),sheet_name=algo[0])
print(df_RI_A,len(df_RI_A.columns)-1)
colors = get_n_colors(len(df_RI_A.columns))
fn = os.path.join(data_path, f"{metric[2]}_{algo[0]}.png")
name = f"{metric[2]} of Column clustering using {algo[0]} clustering"
box_plot(df_RI_A, colors, metric[2],name , fn)"""