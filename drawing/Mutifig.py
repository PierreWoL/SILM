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


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
dataset = "GDS"

labels = ['BERT', 'RoBERTa', 'SBERT']

### Single bar
"""path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/starmie/{dataset}/Sum.xlsx"))
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
        #print(label,label_akk.index, list(label_akk))

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

        plt.tight_layout()

        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1]) + label + f"BarChart.pdf"))
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1]) + label + f"BarChart.png"))
"""

"""def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)

### Long chart PhSAE 1 for all LM
labels = ['BERT', 'RoBERTa', 'SBERT']
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/starmie/{dataset}/Sum.xlsx"))
metrics = ["Purity", "Rand Index"]
width = 0.1

for metric in metrics:
    sum = pd.read_excel(path, sheet_name=metric, index_col=0).iloc[:,].transpose()
    print(sum)
    labels = sum.columns
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(18, 10))
    rects = []
    for num, row_tuple in enumerate(sum.iterrows()):
        index, row = row_tuple
        #print(num, index, "\n", row, "\n", list(row))
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
                             metric + f"BarChartALL_Whole{dataset}.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join(f"result/starmie/{dataset}/"),
                             metric +  f"BarChartALL_Whole{dataset}.png"))
    plt.show()
"""


###The following is for column clustering



"""
LM = ['BERT', 'RoBERTa', 'SBERT']

dataset = "GDS"
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

LM = ['BERT', 'RoBERTa', 'SBERT']
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


### don't know what is it
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

"""
# labels = ['BERT', 'RoBERTa', 'SBERT']

label_per = ['D3L', 'Starmie', 'Pretrain_HI', 'Pretrain_I', 'SILM_SampleCells_HI', 'SILM_SampleCells_I',
             'SILM_SampleCellsTFIDF_HI', 'SILM_SampleCellsTFIDF_I']

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

    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1])  +label + "BarChart.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1])+   label + "BarChart.png"))



plt.tight_layout()
plt.show()
"""

import matplotlib.pyplot as plt


def box(dataset,metrics,num_test = 0.0):
    LM = ['BERT', 'RoBERTa', 'SBERT']

    for metric in metrics:
        print(metrics)
        if metric == "Purity":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/All/P.xlsx"))

        elif metric == "RandIndex":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/All/RandIndex.xlsx"))
        else:
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/{str(num_test)}/TCS.xlsx"))
        df1 = pd.read_excel(path, sheet_name=LM[0], index_col=0).iloc[:-1, ]
        data = [list(df1[i]) for i in df1.columns]

        df2 = pd.read_excel(path, sheet_name=LM[1], index_col=0).iloc[:-1, ]
        data2 = [list(df2[i]) for i in df2.columns]

        df3 = pd.read_excel(path, sheet_name=LM[2], index_col=0).iloc[:-1, ]
        data3 = [list(df3[i]) for i in df3.columns]

        plt.figure(figsize=(18, 10))
        labels = df3.columns
        print(labels)

        cmap = plt.get_cmap('tab10')  # 获取tab10颜色循环，包含10种颜色
        default_colors = cmap.colors  # 获取颜色列表
        colors = default_colors[:8]
        print(colors)

        positions1 = [0.1 + i * 0.2 for i in range(0, 8)]
        bplot = plt.boxplot(data, patch_artist=True, labels=labels, positions=positions1, widths=0.18)

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        positions2 = [positions1[-1] + 0.6 + i * 0.2 for i in range(0, 8)]
        bplot2 = plt.boxplot(data2, patch_artist=True, labels=labels, positions=positions2, widths=0.18)

        for patch, color in zip(bplot2['boxes'], colors):
            patch.set_facecolor(color)

        positions3 = [positions2[-1] + 0.6 + i * 0.2 for i in range(0, 8)]
        bplot3 = plt.boxplot(data3, patch_artist=True, labels=labels, positions=positions3, widths=0.18)

        for patch, color in zip(bplot3['boxes'], colors):
            patch.set_facecolor(color)

        x_position = [positions1[-4], positions2[-4], positions3[-4]]
        x_position_fmt = LM
        plt.xticks([i for i in x_position], x_position_fmt, fontsize=21)

        plt.ylim(0, 1)
        plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1
        plt.ylabel(metric, fontsize=21)
        plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
        plt.legend(bplot['boxes'], labels, loc='upper left', bbox_to_anchor=(0.1, -0.08), fontsize=18,
                   ncol=4)  # 绘制表示框，右下角绘制
        plt.tight_layout()
        if metric == 'TCS':
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/SILM/{dataset}/{str(num_test)}"),
                                     metric + f"BoxPlotALL_Whole{dataset}.pdf"))
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/SILM/{dataset}/{str(num_test)}"),
                                     metric + f"BoxPlot_Whole{dataset}.png"))
        else:

            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/SILM/{dataset}/All"),
                                     metric + f"BoxPlot_Whole{dataset}.pdf"))
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/SILM/{dataset}/All"),
                                     metric + f"BoxPlot_Whole{dataset}.png"))
    plt.show()





def ColumnBar(dataset,metrics,num_test = 0.0):
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=16)

    labels = ['BERT', 'RoBERTa', 'SBERT']
    width = 0.1

    for metric in metrics:

        if metric == "Purity":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/All/P.xlsx"))

        elif metric == "RandIndex":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/All/RandIndex.xlsx"))
        elif metric == "TCS":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/{str(num_test)}/TCS.xlsx"))
        elif metric == "Precision@GroundTruth":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/P4/{dataset}/Attribute.xlsx"))
        elif metric == "Precision":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/P4/{dataset}/Type_P.xlsx"))
        else:
            return 0
        dfs=[]

        for i in labels:
            dfs.append(list(pd.read_excel(path, sheet_name=i, index_col=0).iloc[-1,]))

        sum = pd.DataFrame({ labels[0]: dfs[0], labels[1]: dfs[1], labels[2]: dfs[2]}, index=pd.read_excel(path, sheet_name=labels[0], index_col=0).columns)
        print(sum)


        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(18, 10))
        rects = []
        for num, row_tuple in enumerate(sum.iterrows()):
            index, row = row_tuple
            # print(num, index, "\n", row, "\n", list(row))
            rect = ax.bar(x + (num - 2) * width + 0.015 * num, list(row), width, label=index)
            # rects.append(rect)
            ax.set_ylim(0, 1)
            plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1
            ax.set_ylabel(metric, fontsize=21)
            ax.set_xlabel('Methods', fontsize=21)

            plt.xticks(x, fontsize=21)
            ax.set_xticklabels(labels)
            ax.legend(loc='upper left', bbox_to_anchor=(0.1, -0.08), fontsize=18, ncol=4)
            autolabel(rect)

        fig.tight_layout()
        if metric =='TCS':
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/SILM/{dataset}/{str(num_test)}"),
                                     metric + f"BarChartALL_Whole{dataset}.pdf"))
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/SILM/{dataset}/{str(num_test)}"),
                                     metric + f"BarChartALL_Whole{dataset}.png"))
        elif   metric =="Precision@GroundTruth" or "Precision":
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/P4/{dataset}"),
                                     metric + f"BarChartALL_Whole{dataset}_{metric}.pdf"))
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                     os.path.join(f"result/P4/{dataset}"),
                                     metric + f"BarChartALL_Whole{dataset}_{metric}.png"))
        else:

            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{dataset}/All"),
                                 metric + f"BarChartALL_Whole{dataset}.pdf"))
            plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{dataset}/All"),
                                 metric + f"BarChartALL_Whole{dataset}.png"))
        plt.show()
"""metris = ["Purity", "Rand Index"]
#ColumnBar("GDS",metris)
box("GDS",["TCS"], num_test=0.1)
ColumnBar("GDS",["TCS"], num_test=0.1)
box("WDC",["TCS"], num_test=0.15)
ColumnBar("WDC",["TCS"], num_test=0.15)"""
metris = ["Precision@GroundTruth", "Precision"]
ColumnBar("WDC",metris)
