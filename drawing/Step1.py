import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

barWidth = 0.25

data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                         "result/starmie/TabFact/Subject_Col/summarize.xlsx")
metric = ['Rand Index', 'ARI', 'Purity']
algo = ['Agglomerative', 'BIRCH']

target_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           "result/starmie/TabFact/Subject_Col")


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


for i in range(0, 3):
    metric_index(i)
