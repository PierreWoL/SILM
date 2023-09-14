import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                         "result/starmie/TabFact/Subject_Col/summarize.xlsx")
metric = ['Rand Index', 'Purity']
algo = ['Agglomerative']

target_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           "result/starmie/TabFact/Subject_Col")

Methods = ["SBERT_Instance", "SBERT_HI", "SBERT_SHI", "RoBERTa_Instance", "RoBERTa_HI",
           "RoBERTa_SHI", "SBERT_Instance_SubAttr", "SBERT_HI_SubAttr",
           "RoBERTa_Instance_SubAttr", "RoBERTa_HI_SubAttr", "P_SBERT_Instance",
           "P_SBERT_HI", "P_RoBERTa_Instance", "P_RoBERTa_HI", "P_SBERT_Value", "P_RoBERTa_Value"]

PretrainedMethods = Methods[10:]
Fine_tunedMethods = Methods[:10]


def naming(filename: str):
    filename = filename.split("_metrics.csv")[0]
    methodName = None
    if "Pretrain" in filename:
        if filename.endswith('_True'):
            methodName = PretrainedMethods[4] if 'sbert' in filename else PretrainedMethods[5]
        else:
            filename = filename[:-6]
            if filename.endswith('_none'):
                methodName = PretrainedMethods[0] if 'sbert' in filename else PretrainedMethods[2]
            elif filename.endswith('_header'):
                methodName = PretrainedMethods[1] if 'sbert' in filename else PretrainedMethods[3]
        return methodName
    else:
        if filename.endswith("subCol"):
            filename = filename[:-7]
            if filename.endswith('_none'):
                methodName = Fine_tunedMethods[6] if 'sbert' in filename else Fine_tunedMethods[8]
            elif filename.endswith('_header'):
                methodName = Fine_tunedMethods[7] if 'sbert' in filename else Fine_tunedMethods[9]
        else:
            if filename.endswith('_none'):
                methodName = Fine_tunedMethods[0] if 'sbert' in filename else Fine_tunedMethods[3]
            elif filename.endswith('_subjectheader'):
                methodName = Fine_tunedMethods[2] if 'sbert' in filename else Fine_tunedMethods[5]
            elif filename.endswith('_header'):
                methodName = Fine_tunedMethods[1] if 'sbert' in filename else Fine_tunedMethods[4]
        return methodName


def data_summarize(path):
    fn = {fn: naming(fn) for fn in os.listdir(path) if fn.endswith(".csv") and naming(fn) is not None}
    Results = {'Rand Index': {}, 'Purity': {}}  # 'BIRCH': {},

    for key, v in fn.items():
        result_method = pd.read_csv(os.path.join(path, key),index_col=0)

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
    target_file = os.path.join(path,"summarize.xlsx")
    to_xlsx(RI,purity,file_path=target_file)

import seaborn as sns
def get_n_colors(n):
    return sns.color_palette("husl", n)

#colors = get_n_colors(16)


barWidth = 0.35
def metric_ALL(tar_path, index, embeddingMethods:list, fileName,colors, title,store_path):
    data = pd.read_excel(tar_path, sheet_name=metric[index], index_col=0)
    filtered_data = data[data.index.isin(embeddingMethods)]
    data_sorted = filtered_data.reindex(embeddingMethods)

    print(data_sorted)
    r1 = np.arange(len(data_sorted.iloc[:, 0]))
    plt.figure(figsize=(15, 11))
    #plt.bar(r1, data.iloc[:, 0], color='#FF0088', width=barWidth, edgecolor='white', label=data.columns[0])
    for i, color in zip(r1, colors):
        plt.bar(i, data_sorted.iloc[i, 0], color=color, edgecolor='white', label=data_sorted.index[i])
    plt.xticks([r-barWidth  for r in range(len(data_sorted.iloc[:, 0]))], data_sorted.index)
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


dataset = "TabFact" #TabFact
Table_content = "Subject_Col" #Subject_Col All
subCol_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           os.path.join("result/starmie", dataset, Table_content))
# print(subCol_path)
data_summarize(subCol_path)
colors = [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
          (0.9688417625390765, 0.46710871459052145, 0.1965441952393453),
          (0.8087954113106306, 0.5634700050056693, 0.19502642696727285),
          (0.7008633391290917, 0.6080365980075504, 0.19419512204856468),
          (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
          (0.4225883781014591, 0.677943504931845, 0.19271544738133076),
          (0.19783576093349015, 0.6955516966063037, 0.3995301037444499),
          (0.20518528131112984, 0.6851497738530601, 0.5562527763557912),
          (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
          (0.21576108198845112, 0.6690446872415565, 0.7201192992055431),
          (0.22335772267769388, 0.6565792317435265, 0.8171355503265633),
          (0.3531380715309417, 0.6201408220829481, 0.9586195235634788),
          (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
          (0.8397010947263905, 0.4529020995703274, 0.9578638063653008),
          (0.9603888539940703, 0.3814317878772117, 0.8683117650835491),
          (0.9645179518697552, 0.41602112206844516, 0.708820872610067)]


SUMMARIZE = os.path.join(subCol_path, "summarize.xlsx")
Fine_tunedMethods = Methods[:9] if Table_content == "Subject_Col" else  Methods[:5]

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