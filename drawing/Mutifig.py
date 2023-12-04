import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def table_index(table: pd.DataFrame):
    for index in table.index:
        if 'Pre' in index:
            if 'HI' in index:
                table = table.rename(index={index: 'Pretrain_HI'})
            if 'I' in index:
                table = table.rename(index={index: 'Pretrain_I'})
        elif '_Column' in index:
            if 'SCT' in index and '_HI' in index:
                table = table.rename(index={index: 'SILM_SampleCells_TFIDF_HI'})
            elif 'SCT' in index and '_I' in index:
                table = table.rename(index={index: 'SILM_SampleCells_TFIDF_I'})
            elif re.search("SC\d+", index) and '_HI' in index:
                table = table.rename(index={index: 'SILM_SampleCell_HI'})
            elif re.search("SC\d+", index) and '_I' in index:
                table = table.rename(index={index: 'SILM_SampleCell_I'})
        elif 'sample_cells' in index:
            table = table.rename(index={index: 'Starmie'})

        elif 'D3L' in index:
            table = table.rename(index={index: 'D3L'})
    return table


def table_col(table: pd.DataFrame):
    for col in table.columns:
        if 'Pre' in col:
            if 'HI' in col:
                table = table.rename(columns={col: 'Pretrain_HI'})
            if 'I' in col:
                table = table.rename(columns={col: 'Pretrain_I'})
        elif '_Column' in col:
            if 'SCT' in col and '_HI' in col:
                table = table.rename(columns={col: 'SILM_SampleCells_TFIDF_HI'})
            elif 'SCT' in col and '_I' in col:
                table = table.rename(columns={col: 'SILM_SampleCells_TFIDF_I'})
            elif re.search("SC\d+", col) and '_HI' in col:
                table = table.rename(columns={col: 'SILM_SampleCell_HI'})
            elif re.search("SC\d+", col) and '_I' in col:
                table = table.rename(columns={col: 'SILM_SampleCell_I'})
        elif 'sample_cells' in col:
            table = table.rename(columns={col: 'Starmie'})
        elif 'D3L' in col:
            table = table.rename(columns={col: 'D3L'})
    return table


# labels = ['BERT', 'RoBERTa', 'SBERT']

label_per = ['D3L', 'Starmie', 'Pretrain_HI', 'Pretrain_I', 'SILM_SampleCells_HI', 'SILM_SampleCells_I',
             'SILM_SampleCellsTFIDF_HI', 'SILM_SampleCellsTFIDF_I']
"""
metric = 'Rand Index'
dataset = 'WDC'
data = pd.read_excel(f"D:/CurrentDataset/result/SILM/{dataset}/All/RI.xlsx", index_col=0).iloc[-1:]
data = data[[col for col in data.columns if
             'sample_cells_sbert' not in col and 'sample_cells_roberta' not in col]].transpose()

col_bert = data.loc[[i for i in data.index if '_bert' in i]]
col_sbert = data.loc[[i for i in data.index if '_sbert' in i]]
col_roberta = data.loc[[i for i in data.index if '_roberta' in i]]

value_list = {'BERT': table_index(col_bert), 'RoBERTa': table_index(col_roberta), 'SBERT': table_index(col_sbert)}

for label, values in value_list.items():
    print(values.index,list(values.iloc[:,0]))
    fig, ax = plt.subplots(figsize=(13, 10))
    bars = ax.bar(values.index, list(values.iloc[:,0]), color='skyblue')  # label=values.index
    ax.set_ylim(0, 1)
    plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1

    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=21)
    ax.set_ylabel(metric, fontsize=21)
    ax.set_xlabel('Methods', fontsize=21)
    #ax.legend(loc='upper left', bbox_to_anchor=(0.05, -0.18), fontsize=21)
    ax.set_xticks(np.arange(len(values.index)))
    plt.xticks(fontsize=21)
    ax.set_xticklabels(values.index, rotation=40)
    plt.tight_layout()
"""
"""
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1])  +label + "BarChart.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1])+   label + "BarChart.png"))
"""

"""data = pd.read_excel(f"D:/CurrentDataset/result/SILM/{dataset}/All/RI.xlsx", index_col=0).iloc[:-1,:]
data = data[[col for col in data.columns if
             'sample_cells_sbert' not in col and 'sample_cells_roberta' not in col]]
col_bert = data[[i for i in data.columns if '_bert' in i]]
col_sbert = data[[i for i in data.columns if '_sbert' in i]]
col_roberta = data[[i for i in data.columns if '_roberta' in i]]
value_list = {'BERT': table_col(col_bert), 'RoBERTa': table_col(col_roberta), 'SBERT': table_col(col_sbert)}

"""
# print(col_bert,col_sbert,col_roberta)
"""for label, values in value_list.items():
    plt.figure(figsize=(12, 10))
    colors = ['skyblue']
    bp = values.boxplot(patch_artist=True, notch=False, vert=True, widths=0.5)
    box_colors = colors * (len(values.index))
    boxes = bp.findobj(matplotlib.patches.PathPatch)
    for box, color in zip(boxes, box_colors):
        box.set(facecolor=color)
    plt.xticks(rotation=20, fontsize=21)
    plt.yticks(fontsize=21)
    plt.xlabel('Embedding Methods', fontsize=21)
    plt.ylabel(" ".join(metric.split(" ")[:-1]), fontsize=21)
    plt.subplots_adjust(top=0.9, bottom=0.18)
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
                      for label, color in zip(values.index, box_colors)]

    label = dataset

    plt.ylim(0, 1)


    plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1

    # ax.set_title('Box Plot for Rand Index')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".pdf"))
                             os.path.join(f"result/SILM/{dataset}/All"),
                             " ".join(metric.split(" ")[:-1]) + label + "AttributesBox.pdf"))

    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".png"))
                             os.path.join(f"result/SILM/{dataset}/All"),
                             " ".join(metric.split(" ")[:-1]) + label + "AttributesBox.png"))
    plt.tight_layout()



# plt.savefig("./t9.png")

"""
plt.tight_layout()
plt.show()
"""a = [0.50, 0.80,0.90]
b = [0.37, 0.69,0.100]
c = [0.78, 0.60,0.40]
d = [0.66, 0.86,0.46]
e = [0.80, 0.95,0.67]
# marks = ["o", "X", "+", "*", "O"]



x = np.arange(len(labels))  # 标签位置
width = 0.15  # 柱状图的宽度

fig, ax = plt.subplots()

rects1 = ax.bar(x - width * 2, a, width, label='', hatch="...", color='w', edgecolor="k")
rects2 = ax.bar(x - width + 0.015, b, width, label='b', hatch="oo", color='w', edgecolor="k")
rects3 = ax.bar(x + 0.03, c, width, label='c', hatch="++", color='w', edgecolor="k")
rects4 = ax.bar(x + width + 0.045, d, width, label='d', hatch="XX", color='w', edgecolor="k")
rects5 = ax.bar(x + width * 2 + 0.06, e, width, label='e', hatch="**", color='w', edgecolor="k")


plt.yticks([i / 10.0 for i in range(11)], fontsize=16)  # 设置y轴间隔为0.1
ax.set_ylabel('Rand Index', fontsize=16)
ax.set_xlabel('Embedding Methods', fontsize=16)
#ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""
labels = ['BERT', 'RoBERTa', 'SBERT']

path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/starmie/GDS/Sum_Itself.xlsx"))
metrics = ["Purity", "Rand Index"]
x = np.arange(len(labels))
width = 0.1

for metric in metrics:
    sum = pd.read_excel(path, sheet_name=metric,index_col=0)
    labels = sum.index

    print(labels)
    for label in labels:
        fig, ax = plt.subplots(figsize=(11, 10))
        label_akk = sum.loc[label]
        print(label,label_akk.index, list(label_akk))

        bars = ax.bar(label_akk.index, list(label_akk), color='skyblue', label=label)
        ax.set_ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=21)
        ax.set_ylabel(" ".join(metric.split(" ")[:-1]), fontsize=21)
        ax.set_xlabel('Methods', fontsize=21)

        ax.legend(loc='upper right', fontsize=21)# bbox_to_anchor=(0.05, -0.18)
        plt.xticks(fontsize=21)
        ax.set_xticklabels(label_akk.index, rotation=45)
     
        plt.tight_layout()"""

"""        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1]) + label + f"BarChart.pdf"))
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1]) + label + f"BarChart.png"))"""


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)

### Long chart PhSAE 1 for all LM
labels = ['BERT', 'RoBERTa', 'SBERT']
dataset = 'GoogleSearch'
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/starmie/{dataset}/Sum1.xlsx"))
metrics = ["Purity", "Rand Index"]
width = 0.1

for metric in metrics:
    sum = pd.read_excel(path, sheet_name=metric, index_col=0).iloc[:-1,].transpose()
    labels = sum.columns
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(18, 10))
    rects = []
    for num, row_tuple in enumerate(sum.iterrows()):
        index, row = row_tuple
        print(num, index, "\n", row, "\n", list(row))
        rect = ax.bar(x + (num - 2) * width + 0.015 * num, list(row), width, label=index)
        #rects.append(rect)
        ax.set_ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1
        ax.set_ylabel(metric, fontsize=21)
        ax.set_xlabel('Methods', fontsize=21)

        plt.xticks(x, fontsize=21)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper left',bbox_to_anchor=(0.1, -0.08), fontsize=18,ncol =4)
        autolabel(rect)


    fig.tight_layout()
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join(f"result/starmie/{dataset}/"),
                             metric + "BarChartALL_WholeGDS.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join(f"result/starmie/{dataset}/"),
                             metric +  "BarChartALL_WholeGDS.png"))
    plt.show()



###The following is for column clustering

"""LM = ['BERT', 'RoBERTa']

dataset = "WDC"
metrics = ["Purity", "Rand Index"]
for metric in metrics:
    if metric == "Purity":
        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/P.xlsx"))
    else:
        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/RandIndex.xlsx"))
    for label in LM:
        plt.figure(figsize=(12, 10))
        sum = pd.read_excel(path, sheet_name=label,index_col=0)
        labels = sum.columns
        values = list(sum.iloc[-1,])
        print(values)
        # 创建条形图
        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.bar(labels, values, color='skyblue', label=label)  # color='#FFEBB7', hatch='//',

        # 设置y轴的限制和刻度
        ax.set_ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1

        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=21)

        # 设置标题和坐标轴标签
        ax.set_xlabel('Methods', fontsize=21)
        plt.ylabel(metric, fontsize=21)
        ax.legend(loc='upper left', bbox_to_anchor=(-0.01, -0.18), fontsize=18)
        plt.xticks(fontsize=21)
        ax.set_xticklabels(labels, rotation=30)
        # 显示图形
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".pdf"))
                                 os.path.join(f"result/SILM/{dataset}/All"),
                                 metric + label + "AttributesBarChart.pdf"))

        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".png"))
                                 os.path.join(f"result/SILM/{dataset}/All"),
                                 metric+ label + "AttributesBarChart.png"))

import matplotlib.patches as mpatches

LM = ['BERT', 'RoBERTa']
for metric in metrics:

    if metric == "Purity":

        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/P.xlsx"))
    else:
        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/RandIndex.xlsx"))
    for label in LM:
        plt.figure(figsize=(13, 11))
        df = pd.read_excel(path, sheet_name=label,index_col=0)
        df = df.iloc[:-1,]
        colors = ['skyblue']
        bp = df.boxplot(patch_artist=True, notch=False, vert=True, widths=0.5)
        box_colors = colors * (len(df.columns))
        boxes = bp.findobj(matplotlib.patches.PathPatch)
        for box, color in zip(boxes, box_colors):
            box.set(facecolor=color)
        plt.xticks(rotation=30, fontsize=21)
        plt.yticks(fontsize=21)
        plt.xlabel('Embedding Methods', fontsize=21)
        plt.ylabel(metric, fontsize=21)
        #plt.subplots_adjust(top=0.9, bottom=0.18)
        #legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
           #               for label, color in zip(df.columns, box_colors)]
        legend_patches = [mpatches.Patch(color=color, label=label.format(i + 1)) for i, color in enumerate(colors)]

        plt.legend(handles=legend_patches,loc='upper left', bbox_to_anchor=(-0.01, -0.18), fontsize=18)
        plt.ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1

        # ax.set_title('Box Plot for Rand Index')
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".pdf"))
                                 os.path.join(f"result/SILM/{dataset}/All"),
                                 metric + label + "AttributesBox.pdf"))

        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".png"))
                                 os.path.join(f"result/SILM/{dataset}/All"),
                                 metric + label + "AttributesBox.png"))
        plt.tight_layout()
plt.show()"""