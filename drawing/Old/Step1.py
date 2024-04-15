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
    files = [i[:-12] for i in os.listdir(path) if i.endswith(".csv")]
    print(files)
    fn = {fn: reName(fn) for fn in files if reName(fn) is not None}
    Results = {'Rand Index': {}, 'Purity': {} , 'ACCS':{}}  # 'BIRCH': {},

    for key, v in fn.items():
        result_method = pd.read_csv(os.path.join(path, key+"_metrics.csv"), index_col=0)
        print(result_method)
        Results['Rand Index'][v] = result_method.loc['Random Index', 'Agglomerative']
        Results['Purity'][v] = result_method.loc['Purity', 'Agglomerative']
        #Results['ACCS'][v] = result_method.loc['Average cluster consistency score', 'Agglomerative']

    RI = pd.DataFrame(list(Results['Rand Index'].items()), columns=['Embedding Methods', 'Rand Index'])
    purity = pd.DataFrame(list(Results['Purity'].items()), columns=['Embedding Methods', 'Purity'])
    #ACCS = pd.DataFrame(list(Results['ACCS'].items()), columns=['Embedding Methods', 'Average cluster consistency score'])

    def to_xlsx(df1=None, df2=None, df3=None, file_path='', n1='Rand Index', n2='Purity' ,n3 = 'ACCS'):
        with pd.ExcelWriter(file_path) as writer:
            if df1 is not None:
                df1.to_excel(writer, sheet_name=n1, index=False)
            if df2 is not None:
                df2.to_excel(writer, sheet_name=n2, index=False)
            if df3 is not None:
                df3.to_excel(writer, sheet_name=n3, index=False)
    target_file = os.path.join(path, "summarize.xlsx")
    to_xlsx(RI, purity,  file_path=target_file)#ACCS,


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


dataset = "GDS"  # TabFact GoogleSearch
Table_content = "All"  # Subject_Col All
subCol_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), os.path.join("result/starmie", dataset, Table_content))
#subCol_path= os.path.join(f"C:/Users/1124a/Desktop/Experiments/P1-120/TabFact/{Table_content}")
print(subCol_path)
data_summarize(subCol_path)

SUMMARIZE = os.path.join(subCol_path, f"summarize_{Table_content}.xlsx")
# Fine_tunedMethods = Methods[:9] if Table_content == "Subject_Col" else  Methods[:5]