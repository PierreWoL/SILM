import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from drawing.Utils import morandi_colors
datasets = ["WDC","GDS"]
labels = ['BERT', 'RoBERTa', 'SBERT']
metrics = ["Purity", "Rand Index"]



def P1(metric):
    width = 0.1
    fig, axes = plt.subplots(1, len(datasets), figsize=(40, 10))
    fig.text(0.48, 0.12, 'Methods', fontsize=28)
    # for index, dataset in enumerate(datasets):
    for index, (dataset, ax) in enumerate(zip(datasets, axes)):
        path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                            os.path.join(f"result/starmie/{dataset}/Sum.xlsx"))
        sum = pd.read_excel(path, sheet_name=metric, index_col=0).iloc[:, ].transpose()
        print(sum)
        labels = sum.columns
        x = np.arange(len(labels))
        print("x", x)
        if index == 0:
            ax.set_ylabel(metric, fontsize=36)
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
            index_row, row = row_tuple
            rect = ax.bar(x + (num - 2) * width + 0.015 * num, list(row), width,
                          label=index_row, color=morandi_colors[num], zorder=3)
        # autolabel(ax, rect)

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.12), ncols=8, fontsize=28, ncol=len(sum.index))
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.04, right=0.99, wspace=0.06)
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join(f"result/starmie/"),
                             metric + f"BarChartALL_Whole{metric}.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                             os.path.join(f"result/starmie/"),
                             metric + f"BarChartALL_Whole{metric}.png"))
    plt.show()
metric = metrics[0]
P1(metric)