import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据



# 折线
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           os.path.join("result/starmie/WDC/Sum.xlsx"))
metrics = ["Purity","Rand Index"]


for metric in metrics:
    sum = pd.read_excel(path, sheet_name=metric)
    augmentation_times = sum["Augmentation Times"]
    sample_cells_tfidf_sbert = sum["Sample_cells_TFIDF_SBERT"]
    sample_cells_sbert = sum["Sample_cells_SBERT"]
    sample_cells_tfidf_roberta = sum["Sample_cells_TFIDF_RoBERTa"]
    sample_cells_roberta = sum["Sample_cells_RoBERTa"]
    sample_cells_tfidf_bert = sum["Sample_cells_TFIDF_BERT"]
    sample_cells_bert = sum["Sample_cells_BERT"]

    # 绘制
    plt.figure(figsize=(10, 6))
    #
    #
    plt.plot(augmentation_times, sample_cells_tfidf_roberta, marker='o', label='Sample_cells_TFIDF_RoBERTa')
    plt.plot(augmentation_times, sample_cells_roberta, marker='o', label='Sample_cells_RoBERTa')
    plt.plot(augmentation_times, sample_cells_tfidf_bert, marker='o', label='Sample_cells_TFIDF_BERT')
    plt.plot(augmentation_times, sample_cells_bert, marker='o', label='Sample_cells_BERT')
    plt.plot(augmentation_times, sample_cells_tfidf_sbert, marker='o', label='Sample_cells_TFIDF_SBERT')
    plt.plot(augmentation_times, sample_cells_sbert, marker='o', label='Sample_cells_SBERT')

    # 标题和轴标签

    plt.xlabel('Augmentation Times', fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.xticks(fontsize=16)

    plt.ylim(0, 1)  # 设置y轴范围为0到1
    plt.yticks([i / 10.0 for i in range(11)], fontsize=16)  # 设置y轴间隔为0.1
    legend = plt.legend(loc='lower center', ncol=2, fontsize=16)  # 显示图例，位置在下方并分为4列
    legend.get_frame().set_edgecolor('none')  # 为图例去掉边框
    legend.get_frame().set_alpha(0)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                           os.path.join("result/starmie/WDC/"), metric+"LineChart.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join("result/starmie/WDC/"), metric + "LineChart.png"))
    plt.show()




"""
# 条形-- Phase 1 -- 单条
datasets = ["WDC", "TabFact"]
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"C:/Users/1124a/Desktop/Sum.xlsx"))#result/starmie/WDC/Sum.xlsx
metrics = ["Purity C", "Rand Index C"]
for index, dataset in enumerate(datasets):

    for metric in metrics:
        sum = pd.read_excel(path, sheet_name=metric)
        labels = sum.columns[1:-1]

        values = list(sum.iloc[index,])[1:-1]
        print(values)
        label = list(sum.iloc[index,])[0]
        # 创建条形图
        fig, ax = plt.subplots(figsize=(11, 10))

        bars = ax.bar(labels, values, color='skyblue', label=label)  if index ==0 else \
            ax.bar(labels, values, color='#FFEBB7', hatch='//',label=label)
        #

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
        ax.set_ylabel(" ".join(metric.split(" ")[:-1]), fontsize=21)
        ax.set_xlabel('Methods', fontsize=21)

        ax.legend(loc='upper left', bbox_to_anchor=(0.05, -0.18), fontsize=21)
        plt.xticks(fontsize=21)
        ax.set_xticklabels(labels, rotation=20)
        # 显示图形
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1])  +label + "BarChart.pdf"))
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/starmie/{dataset}/"),
                                 " ".join(metric.split(" ")[:-1])+   label + "BarChart.png"))
"""

"""
metrics = ["Purity C", "Rand Index C"]
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/starmie/WDC/Sum.xlsx"))
# 条形-- Phase 1 -- 双条
for metric in metrics:
    sum = pd.read_excel(path, sheet_name=metric)
    labels = sum.columns[1:-1]
    print(labels)
    # labels = ['SC5_sbert_HI', 'SC5_sbert_I', 'SCT5_sbert_HI', 'SCT5_sbert_I', 'Pre_sbert_HI', 'Pre_sbert_I', 'D3L']
    # values = [0.812, 0.717, 0.802, 0.804, 0.684, 0.649, 0.493]
    wdc_values = list(sum.iloc[0,])[1:-1]
    tabfact_values = list(sum.iloc[1,])[1:-1]

    bar_width = 0.25
    gap_within_group = 0.06
    gap_between_groups = 0.2
    positions = np.arange(0, len(labels) * (bar_width * 2 + gap_within_group + gap_between_groups),
                          bar_width * 2 + gap_within_group + gap_between_groups)[:-1]
    print(positions)
    r1 = np.arange(len(wdc_values))
    r2 = [x + bar_width + 0.2 for x in r1]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars1 = ax.bar(positions, wdc_values, width=bar_width, color="skyblue",  # '#1F77B4'
                   label="WDC")  # Smaller Real的颜色 list(sum.iloc[0,])[0]
    bars2 = ax.bar(positions + bar_width + gap_within_group, tabfact_values, width=bar_width, color='#FFEBB7',
                   hatch='//', label="TabFact-Wiki")  # Synthetic的颜色和纹理 list(sum.iloc[1,])[0]

    # 设置y轴的限制和刻度
    ax.set_ylim(0, 1)
    plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1

    for bar in bars1:
        height = bar.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 3, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=21)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 4 * 3, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=21)

    # 设置标题和坐标轴标签
    ax.set_ylabel(" ".join(metric.split(" ")[:-1]), fontsize=21)
    # 确定当前y轴的上下限制

    ax.set_xlabel('Methods', fontsize=21)
    ax.set_xticks(positions + bar_width / 2)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, -0.18), fontsize=18)
    # ax.legend(frameon=False, loc='upper left', fontsize=21)
    plt.xticks(fontsize=21)

    # 显示图形
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join("result/starmie/WDC/"), metric + "_AllBarChart.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join("result/starmie/WDC/"), metric + "_AllBarChart.png"))
    plt.show()
"""
"""
#条形-- Phase 2 -- 单条

dataset = "TabFact"
metrics = ["Purity C", "Rand Index C"]
for metric in metrics:
    if metric == "Purity C":

        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/Purity.xlsx"))
    else:
        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/RI.xlsx"))
    sum = pd.read_excel(path, sheet_name="Agglomerative")
    labels = sum.columns[1:]
    # labels = ['SC5_sbert_HI', 'SC5_sbert_I', 'SCT5_sbert_HI', 'SCT5_sbert_I', 'Pre_sbert_HI', 'Pre_sbert_I', 'D3L']
    # values = [0.812, 0.717, 0.802, 0.804, 0.684, 0.649, 0.493]
    values = list(sum.iloc[-1,])[1:]
    print(values)
    label = dataset  # list(sum.iloc[-1,])[0]
    # 创建条形图
    fig, ax = plt.subplots(figsize=(11, 10))
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
    ax.set_ylabel(" ".join(metric.split(" ")[:-1]), fontsize=21)
    ax.set_xlabel('Methods', fontsize=21)

    ax.legend(loc='upper left', bbox_to_anchor=(0.05, -0.18), fontsize=21)
    plt.xticks(fontsize=21)
    ax.set_xticklabels(labels, rotation=20)
    # 显示图形
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".pdf"))
                             os.path.join(f"result/SILM/{dataset}/All"),
                             " ".join(metric.split(" ")[:-1]) + label + "Attributes.pdf"))

    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".png"))
                             os.path.join(f"result/SILM/{dataset}/All"),
                             " ".join(metric.split(" ")[:-1]) + label + "Attributes.png"))



# 条形-- Phase 2 --  box plot
metrics = ["Purity C", "Rand Index C"]
for metric in metrics:
    plt.figure(figsize=(11, 10))
    if metric == "Purity C":

        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/Purity.xlsx"))
    else:
        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/SILM/{dataset}/All/RI.xlsx"))
    df = pd.read_excel(path, sheet_name="Agglomerative")
    df = df.iloc[:-1, 1:]
    colors = ['skyblue']
    bp = df.boxplot(patch_artist=True, notch=False, vert=True, widths=0.5)
    box_colors = colors * (len(df.columns))
    boxes = bp.findobj(matplotlib.patches.PathPatch)
    for box, color in zip(boxes, box_colors):
        box.set(facecolor=color)
    plt.xticks(rotation=20, fontsize=21)
    plt.yticks(fontsize=21)
    plt.xlabel('Embedding Methods', fontsize=21)
    plt.ylabel(" ".join(metric.split(" ")[:-1]), fontsize=21)
    plt.subplots_adjust(top=0.9, bottom=0.18)
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
                      for label, color in zip(df.columns, box_colors)]
    # bp = plt.boxplot(table.values, labels=table.columns, patch_artist=True,notch=True,showfliers=False)
    # fig, ax = plt.subplots(figsize=(11, 9))
    label = dataset
    # bp = ax.boxplot(df.values, patch_artist=True, vert=True, notch=False)
    # for patch, color in zip(bp['boxes'], colors * (len(df.columns))): #, h  , hatch * (len(df.columns) // 2)
    # patch.set_facecolor(color)
    # patch.set_hatch(h)
    # plt.yticks( fontsize=16)  # 设置y轴间隔为0.1
    # ax.set_ylabel(" ".join(metric.split(" ")[:-1]), fontsize=16)
    # ax.set_ylim(0.5, 1)  # 设置y轴的范围

    # ax.set_xlabel('Methods', fontsize=16)

    # ax.set_xticklabels(df.columns, rotation=20, ha="right", fontsize=16) 
    plt.ylim(0, 1)
    plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1

    # ax.set_title('Box Plot for Rand Index')
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".pdf"))
                             os.path.join(f"result/SILM/{dataset}/All"), " ".join(metric.split(" ")[:-1]) +label+ "AttributesBox.pdf"))

    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".png"))
                             os.path.join(f"result/SILM/{dataset}/All"), " ".join(metric.split(" ")[:-1])  +label+ "AttributesBox.png"))
    plt.tight_layout()


"""

"""


dataset = "TabFact"

path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/SILM/{dataset}/Agglomerative_overall_tcs.csv"))
df = pd.read_csv(path)
df = df.iloc[:-1, 2:]
plt.figure(figsize=(11, 10))
colors = ['skyblue']  # 根据您给出的示例，您可以根据需要更改这些颜色
bp = df.boxplot(patch_artist=True, notch=False, vert=True, widths=0.5)
box_colors = colors * (len(df.columns))

boxes = bp.findobj(matplotlib.patches.PathPatch)
for box, color in zip(boxes, box_colors):
    box.set(facecolor=color)
plt.xticks(rotation=20, fontsize=21)

plt.xlabel('Embedding Methods', fontsize=21)
plt.ylim(0, 1)
plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1
plt.ylabel("Tree Consistency Score", fontsize=21)

plt.subplots_adjust(top=0.9, bottom=0.18)
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
                  for label, color in zip(df.columns, box_colors)]

#fig, ax = plt.subplots(figsize=(11, 9))

#label = "WDC"
#color = ['skyblue']
#colors = color * (len(df.columns))
#bp = df.boxplot(patch_artist=True, notch=False, vert=True,)
#bp = ax.boxplot(df.values, patch_artist=False, vert=True, notch=False)

# 设置箱子的颜色和填充样式
 # 根据您给出的示例，您可以根据需要更改这些颜色
# hatch = ['//']

#plt.yticks(fontsize=16)  # 设置y轴间隔为0.1
#ax.set_ylabel("Tree Consistency Score", fontsize=16)
#ax.set_ylim(0.5, 1)  # 设置y轴的范围

#ax.set_xlabel('Methods', fontsize=16)

#ax.set_xticklabels(df.columns, rotation=20, ha="right", fontsize=16)

# ax.set_title('Box Plot for Rand Index')
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                         # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".pdf"))
                         os.path.join(f"result/SILM/{dataset}/"), f"TCS{dataset}_box.pdf"))

plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                         # os.path.join("result/starmie/WDC/"), " ".join(metric.split(" ")[:-1]) +label+ ".png"))
                         os.path.join(f"result/SILM/{dataset}/"), f"TCS{dataset}_box.png"))
plt.tight_layout()

# SILM tree consistency metrics


path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    os.path.join(f"result/SILM/{dataset}/Sum.xlsx"))
sum = pd.read_excel(path, sheet_name="Sheet1")
labels = sum.columns

values = list(sum.iloc[0,])
# 创建条形图
fig, ax = plt.subplots(figsize=(11, 10))
bars = ax.bar(labels, values, color='skyblue', label="WDC")

ax.set_ylim(0, 1)
plt.yticks([i / 10.0 for i in range(11)], fontsize=21)  # 设置y轴间隔为0.1
#bars = ax.bar(labels, values, color='skyblue', label=label)  if index ==0 else \
       #     ax.bar(labels, values, color='#FFEBB7', hatch='//',label=label)
for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=21)

        # 设置标题和坐标轴标签
ax.set_ylabel("Tree Consistency Score", fontsize=21)
ax.set_xlabel('Methods', fontsize=21)
ax.legend(loc='upper left', bbox_to_anchor=(0.05, -0.18), fontsize=21)
plt.xticks(fontsize=21)
ax.set_xticklabels(labels, rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{dataset}/"), f"TCSbar_{dataset}.pdf"))
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{dataset}/"),f"TCSbar_{dataset}.png"))
plt.show()





"""