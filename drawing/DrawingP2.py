import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from drawing.Utils import morandi_colors

LMs = ['BERT', 'RoBERTa', 'SBERT']


def ColumnBar(datasets, metric, num_test=0.0):
    width = 0.1
    fig, axes = plt.subplots(1, len(datasets), figsize=(40, 11))
    fig.text(0.48, 0.12, 'Methods', fontsize=28)
    for index, (dataset, ax) in enumerate(zip(datasets, axes)):
        rects = []
        if metric == "Purity":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/All/P.xlsx"))
        elif metric == "Rand Index":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/All/RandIndex.xlsx"))
        elif metric == "TCS":
            path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                os.path.join(f"result/SILM/{dataset}/{str(num_test)}/TCS.xlsx"))
        else:
            return 0
        dfs = []
        for i in LMs:
            dfs.append(list(pd.read_excel(path, sheet_name=i, index_col=0).iloc[-1,]))

        sum = pd.DataFrame({LMs[0]: dfs[0], LMs[1]: dfs[1], LMs[2]: dfs[2]},
                           index=pd.read_excel(path, sheet_name=LMs[0], index_col=0).columns)
        print(sum)
        labels = sum.columns
        x = np.arange(len(labels))
        metric_label = metric
        if index == 0:
            if metric == "TCS":
                metric_label = "Tree Consistency Score"
            ax.set_ylabel(metric_label, fontsize=36)
        ax.set_ylim(0, 1)
        yticks = [i / 10.0 for i in range(11)]
        ax.set_yticks(yticks)
        for y in yticks:
            ax.axhline(y=y, color='gray', linewidth=0.5, linestyle='-')
        ax.tick_params(axis='y', labelsize=28)
        ax.set_xticks(x)  # 设置x轴刻度
        ax.set_xticklabels(labels, fontsize=28)  # 设置x轴标签
        ax.set_title(dataset, fontsize=36)
        rects = []
        for num, row_tuple in enumerate(sum.iterrows()):
            index, row = row_tuple
            # print(num, index, "\n", row, "\n", list(row))
            rect = ax.bar(x + (num - 2) * width + 0.018 * num, list(row),
                          width, label=index, color=morandi_colors[num], zorder=3)

        os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                     os.path.join(f"result/SILM/{str(num_test)}" + metric + f"BarChart_Whole_TCS.pdf"))
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), ncols=8, fontsize=28, ncol=len(sum.index))
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.04, right=0.99, wspace=0.06)
    if metric == 'TCS':
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{str(num_test)}" + metric + f"BarChart_Whole_TCS.pdf")))
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{str(num_test)}" + metric + f"BarChart_Whole_TCS.png")))
    else:
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{metric}_BarChart_Whole.pdf")))
        plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 os.path.join(f"result/SILM/{metric}_BarChart_Whole.png")))
    plt.show()


data = ["WDC", "GDS"]
metrics = ["Purity", "Rand Index"]
for me in metrics:
    ColumnBar(data, me)

ColumnBar(data, "TCS", num_test=0.15)
