from Step2_1 import box_plot,algo
from Step1 import naming, colors
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def TreeConsistency():
    box_colorsM = colors[:5] + colors[10:]
    for algorithm in algo:
        overall_tcs = {}
        data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "result/Valerie/WDC")
        folders = [fn for fn in os.listdir(data_path) if "." not in fn and "_" in fn and 'sub']
        for index, fold in enumerate(folders):
            name_cols = naming(fold)
            print(name_cols, fold)
            table = pd.read_csv(os.path.join(data_path, fold, "TreeConsistencyScore.csv"), index_col=0)
            table = table[table['ClusteringAlgorithm'].str.contains(algorithm)]
            table = table.sort_index()
            if index == 0:
                overall_tcs["label"] = table.iloc[:, 0]

            overall_tcs[name_cols] = table.iloc[:, 1]

        df2 = pd.DataFrame(overall_tcs)
        new_df = df2.set_index('label')
        df2.to_csv(os.path.join(data_path, algorithm+"_overall_tcs.csv"))

        y_name = "Tree Consistency Score"
        title = "Tree Consistency Score of Embedding Methods of %s Clustering Algorithm" %algorithm
        fn = os.path.join(data_path, algorithm+"_overall_tcs.png")
        box_plot(df2, box_colorsM, y_name, title, fn)


TreeConsistency()


def Layer_Purity(i):
    lost = []
    print(i)
    datas = []
    methods = []
    data_p = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "result/Valerie/TabFact", str(i),
                          'Agglomerative')
    folders = [fn for fn in os.listdir(data_p) if "." not in fn]
    for fold in folders:
        naming_embed = naming(fold)
        if os.path.exists(os.path.join(data_p, fold, "layer_purity.csv")):
            methods.append(naming_embed)
            purity = pd.read_csv(os.path.join(data_p, fold, "layer_purity.csv"))
            purity = purity[['layer', 'Purity']]
            datas.append(purity)
        else:
            lost.append(naming_embed)
    if len(datas)>0:
        result = pd.concat([df.set_index('layer') for df in datas], axis=1)
        result = result.sort_values(by='layer')
        result.columns = methods
        # print(result)
        result.to_csv(os.path.join(data_p, "all_purity.csv"))
        return True
    else:
        return False


file_n =  os.listdir(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),"result/Valerie/TabFact"))
id_files = [i for i in file_n if "_" not in i]
def metric_index():
    barWidth = 0.13
    Example_index = id_files
    for i in Example_index:
        if Layer_Purity(i) is True:
            print("aHa")
            data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "result/Valerie/TabFact", str(i),
                                     'Agglomerative')
            csv_file = os.path.join(data_path, "all_purity.csv")
            data = pd.read_csv(csv_file)
            print(data)
            LAYERS = [f'Layer {i}' for i in range(0, len(data))]
            plt.figure(figsize=(10, 6))
            for xi in range(1, len(data.columns)):
                r2 = [x + (xi - 1) * barWidth for x in np.arange(len(data.iloc[:, 0]))]
                print(data.iloc[:, xi])
                plt.bar(r2, data.iloc[:, xi], color=colors[xi - 1], width=barWidth, edgecolor='white',
                        label=data.columns[xi])
            plt.xticks([r + barWidth for r in range(len(data.iloc[:, 0]))], LAYERS)
            plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
            plt.xticks(rotation=20, fontsize=11)
            plt.yticks(fontsize=11)
            plt.xlabel('Embedding Methods', fontsize=10)
            plt.title(f"Layer Purity for CLUSTER {i} using Different Embedding Methods ", fontsize=14)
            plt.ylabel("Purity", fontsize=11)
            plt.subplots_adjust(top=0.9, bottom=0.12, right=0.77, left=0.1)
            fn = os.path.join(data_path, f"{i}_LayerPurity.png")
            plt.savefig(fn)
            fn2 = os.path.join(data_path, f"{i}_LayerPurity_Box.png")
            box_plot(data.iloc[:, 1:], colors, "Purity",
                     f"Layer Purity for CLUSTER {i} using Different Embedding Methods ", fn2)
            plt.show()





def box():
    # data是acc中三个箱型图的参数
    data = [
        [0.8676, 0.8484, 0.8293, 0.8917, 0.9151, 0.9470, 0.8935, 0.8078, 0.9081, 0.8555, 0.8897, 0.9062, 0.9190, 0.8964,
         0.8520, 0.8697, 0.8738],
        [0.8512, 0.8026, 0.7911, 0.8787, 0.9131, 0.9532, 0.8656, 0.8159, 0.9187, 0.8421, 0.8758, 0.9096, 0.9128, 0.8951,
         0.8748, 0.8537, 0.8750],
        [0.9161, 0.9047, 0.8635, 0.9026, 0.9328, 0.9490, 0.8911, 0.8669, 0.9227, 0.8683, 0.9114, 0.9372, 0.9475, 0.9053,
         0.8839, 0.9364, 0.9032]]
    # data2 是F1 score中三个箱型图的参数
    data2 = [
        [0.9291, 0.9180, 0.9067, 0.9427, 0.9557, 0.9728, 0.9438, 0.8937, 0.9518, 0.9221, 0.9416, 0.9508, 0.9578, 0.9454,
         0.9201, 0.9303, 0.9327],
        [0.9196, 0.8905, 0.8834, 0.9354, 0.9546, 0.9760, 0.9279, 0.8986, 0.9576, 0.9143, 0.9338, 0.9527, 0.9544, 0.9447,
         0.9332, 0.9211, 0.9333],
        [0.9562, 0.9500, 0.9267, 0.9488, 0.9652, 0.9738, 0.9424, 0.9287, 0.9598, 0.9295, 0.9536, 0.9676, 0.9731, 0.9503,
         0.9384, 0.9672, 0.9491]]
    # data3 是IoU中三个箱型图的参数
    data3 = [
        [0.8733, 0.8624, 0.8673, 0.8815, 0.9363, 0.9433, 0.9163, 0.8350, 0.9094, 0.8878, 0.8956, 0.9050, 0.9238, 0.9077,
         0.8686, 0.8747, 0.8877],
        [0.8563, 0.8368, 0.8618, 0.8743, 0.9406, 0.9479, 0.8866, 0.8473, 0.9195, 0.8679, 0.8922, 0.9091, 0.9225, 0.9111,
         0.8857, 0.8629, 0.8910],
        [0.9172, 0.9091, 0.8864, 0.9029, 0.9503, 0.9530, 0.9200, 0.8857, 0.9211, 0.9033, 0.9201, 0.9391, 0.9430, 0.9227,
         0.9056, 0.9360, 0.9145]]
    # 箱型图名称
    labels = ["A", "B", "C"]
    # 三个箱型图的颜色 RGB （均为0~1的数据）
    colors = [(202 / 255., 96 / 255., 17 / 255.), (255 / 255., 217 / 255., 102 / 255.),
              (137 / 255., 128 / 255., 68 / 255.)]
    # 绘制箱型图
    # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
    bplot = plt.boxplot(data, patch_artist=True, labels=labels, positions=(1, 1.4, 1.8), widths=0.3)
    # 将三个箱分别上色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(data2, patch_artist=True, labels=labels, positions=(2.5, 2.9, 3.3), widths=0.3)

    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(data3, patch_artist=True, labels=labels, positions=(4, 4.4, 4.8), widths=0.3)

    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    x_position = [1, 2.5, 4]
    x_position_fmt = ["acc", "F1 score", "IoU"]
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    plt.ylabel('percent (%)')
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'], labels, loc='lower right')  # 绘制表示框，右下角绘制
    plt.savefig(fname="pic.png", figsize=[10, 10])
    plt.show()
