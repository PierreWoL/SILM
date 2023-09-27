import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Step2_1 import reName

data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                         "result/starmie/TabFact/Subject_Col/summarize.xlsx")
metric = ['Rand Index', 'Purity']
algo = ['Agglomerative']

target_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           "result/starmie/TabFact/Subject_Col")

test = "cl_sample_cells_lm_sbert_head_column_0_subjectheader_subCol_metrics.csv"


def data_summarize(path):
    fn = {fn: reName(fn.split("_metrics.csv")[0]) for fn in os.listdir(path) if
          fn.endswith(".csv") and reName(fn) is not None}
    Results = {'Rand Index': {}, 'Purity': {}}  # 'BIRCH': {},

    for key, v in fn.items():
        result_method = pd.read_csv(os.path.join(path, key), index_col=0)

        Results['Rand Index'][v] = result_method.loc['random Index', 'Agglomerative']
        Results['Purity'][v] = result_method.loc['purity', 'Agglomerative']
    RI = pd.DataFrame(list(Results['Rand Index'].items()), columns=['Embedding Methods', 'Rand Index'])
    purity = pd.DataFrame(list(Results['Purity'].items()), columns=['Embedding Methods', 'purity'])

    def to_xlsx(df1=None, df2=None, file_path='', n1='Rand Index', n2='Purity'):
        with pd.ExcelWriter(file_path) as writer:
            if df1 is not None:
                df1.to_excel(writer, sheet_name=n1, index=False)
            if df2 is not None:
                df2.to_excel(writer, sheet_name=n2, index=False)

    target_file = os.path.join(path, "summarize.xlsx")
    to_xlsx(RI, purity, file_path=target_file)


# colors = get_n_colors(16)


barWidth = 0.35


def metric_ALL(tar_path, index, embeddingMethods: list, fileName, colors, title, store_path):
    data = pd.read_excel(tar_path, sheet_name=metric[index], index_col=0)
    filtered_data = data[data.index.isin(embeddingMethods)]
    data_sorted = filtered_data.reindex(embeddingMethods)

    print(data_sorted)
    r1 = np.arange(len(data_sorted.iloc[:, 0]))
    plt.figure(figsize=(15, 11))
    # plt.bar(r1, data.iloc[:, 0], color='#FF0088', width=barWidth, edgecolor='white', label=data.columns[0])
    for i, color in zip(r1, colors):
        plt.bar(i, data_sorted.iloc[i, 0], color=color, edgecolor='white', label=data_sorted.index[i])
    plt.xticks([r - barWidth for r in range(len(data_sorted.iloc[:, 0]))], data_sorted.index)
    # plt.legend()
    plt.xticks(rotation=45, fontsize=12)
    # plt.yticks(fontsize=11)
    plt.xlabel('Embedding Methods', fontsize=12)
    plt.title(f"{metric[index]} of Identifying top-level types using Agglomerative Clustering --{title} ", fontsize=18)
    plt.ylabel(metric[index], fontsize=12)
    plt.subplots_adjust(top=0.9, bottom=0.19)
    fn = os.path.join(store_path, f"{fileName}_{metric[index]}_Step1.pdf")
    plt.savefig(fn)
    plt.show()


dataset = "WDC"  # TabFact
Table_content = "Subject_Col"  # Subject_Col All
subCol_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           os.path.join("result/starmie", dataset, Table_content))
# print(subCol_path)
data_summarize(subCol_path)

SUMMARIZE = os.path.join(subCol_path, f"summarize_{Table_content}.xlsx")
# Fine_tunedMethods = Methods[:9] if Table_content == "Subject_Col" else  Methods[:5]

"""


for i in [0,1]:

    if  Table_content != "Subject_Col":
        color1 = colors[0:]
        metric_ALL(SUMMARIZE, i, Fine_tunedMethods+PretrainedMethods, dataset + Table_content + "_all", colors[:5]+colors[10:], "All Methods",subCol_path)

        metric_ALL(SUMMARIZE, i, PretrainedMethods, dataset + Table_content + "_Pretrain", colors[10:],
                   "Pretrained Methods",subCol_path)
        metric_ALL(SUMMARIZE, i, Fine_tunedMethods, dataset + Table_content + "_FT", colors[:5], "Fine-tuned Methods",subCol_path)
    else:

        metric_ALL(SUMMARIZE, i, Methods, dataset + Table_content + "_all", colors, "All Methods",subCol_path)

        metric_ALL(SUMMARIZE, i, PretrainedMethods, dataset + Table_content + "_Pretrain", colors[10:],
                   "Pretrained Methods",subCol_path)
        metric_ALL(SUMMARIZE, i, Fine_tunedMethods, dataset + Table_content + "_FT", colors[:9], "Fine-tuned Methods",subCol_path)

"""

"""




def metric_index(index):
    data = pd.read_excel(data_path, sheet_name=metric[index], index_col=0)
    print(data)
    r1 = np.arange(len(data.iloc[:, 0]))
    r2 = [x + barWidth for x in r1]
    EMBEDMETHODS = ['SBERT_none', 'RoBERTa_none', 'SBERT_subjectheader', 'RoBERTa_subjectheader', 'SBERT_header',
                    'RoBERTa_header']
    plt.figure(figsize=(8, 6))
    plt.bar(r1, data.iloc[:, 0], color='#FF0088', width=barWidth, edgecolor='white', label=data.columns[0])
    plt.bar(r2, data.iloc[:, 1], color='#00BBFF', width=barWidth, edgecolor='white', label=data.columns[1])
    plt.xticks([r + barWidth for r in range(len(data.iloc[:, 0]))], EMBEDMETHODS)
    plt.legend()
    plt.xticks(rotation=20, fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel('Embedding Methods', fontsize=10)
    plt.title(f"{metric[index]} of Table Clustering ", fontsize=14)
    plt.ylabel(metric[index], fontsize=10)
    plt.subplots_adjust(top=0.9, bottom=0.19)
    fn = os.path.join(target_path, f"{metric[index]}_Step1.png")
    plt.savefig(fn)
    plt.show()
"""
