import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experimentalData import get_files
import os
import mplcursors
from Utils import mkdir

"""
Method3_MODE2Sim0.3DistributionLSH_metrics
# Create a sample dataframe
data = {'year': [2010, 2011, 2012, 2013, 2014, 2015],
        'line1': [10, 15, 13, 17, 20, 25],
        'line2': [5, 10, 8, 12, 15, 20],
        'line3': [20, 25, 23, 27, 30, 35]}
df = pd.DataFrame(data)

# Plot the line chart
plt.plot(df['year'], df['line1'], label='Line 1')
plt.plot(df['year'], df['line2'], label='Line 2')
plt.plot(df['year'], df['line3'], label='Line 3')

# Add chart title and axis labels
plt.title('Multi-Line Chart')
plt.xlabel('Year')
plt.ylabel('Value')

# Add legend
plt.legend()

# Show the chart
plt.show()
"""
DATA_PATH = ['open_data', 'SOTAB', 'Test_corpus', 'T2DV2']
absolute_path = os.getcwd() + "/result/metrics/"
absolute_path_starmie = os.getcwd() + "/result/starmie/"
kinds = ["Distribution", "Value", "Format", "Header", "Embed"]


def index_drawing():
    for kind in kinds:
        BIRCH_RI, Hierarchical_RI = {}, {}
        BIRCH_Purity, Hierarchical_Purity = {}, {}
        for dataset in DATA_PATH:
            BIRCH_RI[dataset] = []
            Hierarchical_RI[dataset] = []
            BIRCH_Purity[dataset] = []
            Hierarchical_Purity[dataset] = []
            result = absolute_path + dataset + "/Method3/MODE2/"
            files = get_files(result)
            print(files)
            column = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for file in files:
                file_last = file.lstrip("Method3_MODE2Sim")
                if kind in file_last or kind.lower() in file_last:
                    # print(file_last,result + file + ".csv" )
                    data = pd.read_csv(result + file + ".csv", header=0, index_col=0)
                    # print( data.iloc[3, 0],data.iloc[3, 1],data.iloc[6, 0],data.iloc[6, 1])
                    BIRCH_RI[dataset].append(data.iloc[3, 0])
                    Hierarchical_RI[dataset].append(data.iloc[3, 1])
                    BIRCH_Purity[dataset].append(data.iloc[6, 0])
                    Hierarchical_Purity[dataset].append(data.iloc[6, 1])
        df_BIRCH_RI = pd.DataFrame.from_dict(BIRCH_RI, orient='index', columns=column)
        df_Hierarchical_RI = pd.DataFrame.from_dict(Hierarchical_RI, orient='index', columns=column)
        df_BIRCH_Purity = pd.DataFrame.from_dict(BIRCH_Purity, orient='index', columns=column)
        df_Hierarchical_Purity = pd.DataFrame.from_dict(Hierarchical_Purity, orient='index', columns=column)
        dfs = {'BIRCH Rand Index': df_BIRCH_RI,
               'HierarchicalClustering Rand Index': df_Hierarchical_RI,
               'BIRCH Purity': df_BIRCH_Purity,
               'HierarchicalClustering Purity': df_Hierarchical_Purity}
        print(df_BIRCH_RI, "\n", df_Hierarchical_RI, "\n", df_BIRCH_Purity, "\n", df_Hierarchical_Purity)
        for key, df in dfs.items():
            print(key, df)

            for data_path in DATA_PATH:
                plt.plot(df.columns, df.loc[data_path], label=data_path)
            # Add chart title and axis labels
            plt.title(key + " Tuning Similarity threshold in \n" + kind + " LSH indexes")
            plt.xlabel('Similarity threshold')
            plt.ylabel(key.split(' ', 1)[1])
            # Add legend
            plt.legend()
            plt.savefig("fig/" + kind + key + ".png")
            # Show the chart
            plt.show()

            plt.close()


def dfs_sum(is_sub: tuple):
    for sub in is_sub:
        BIRCH_RI = {}
        BIRCH_Purity = {}
        Hierarchical_RI = {}
        Hierarchical_Purity = {}
        dfs = {}
        for dataset in DATA_PATH:
            dataset_BIRCH_RI = {}
            dataset_BIRCH_P = {}
            dataset_Hierarchical_RI = {}
            dataset_Hierarchical_P = {}
            data_path = os.path.join(absolute_path_starmie, dataset, sub)
            files = [fn for fn in os.listdir(data_path)]
            files.sort()
            for file in files:
                print(file)
                result = pd.read_csv(os.path.join(data_path, file), header=0, index_col=0)
                method = sub + file[:-4]
                dataset_BIRCH_RI[method] = result.iloc[3, 0]
                dataset_Hierarchical_RI[method] = result.iloc[3, 1]
                dataset_BIRCH_P[method] = result.iloc[6, 0]
                dataset_Hierarchical_P[method] = result.iloc[6, 1]
                # print(result,"\n",dataset_BIRCH_RI,dataset_Hierarchical_RI,dataset_BIRCH_P,dataset_Hierarchical_P)
            BIRCH_RI[dataset] = dataset_BIRCH_RI
            BIRCH_Purity[dataset] = dataset_BIRCH_P
            Hierarchical_RI[dataset] = dataset_Hierarchical_RI
            Hierarchical_Purity[dataset] = dataset_Hierarchical_P

        df_BIRCH_RI = pd.DataFrame.from_dict(BIRCH_RI, orient='index')
        dfs["BIRCH_RI"] = df_BIRCH_RI.transpose()
        df_Hierarchical_RI = pd.DataFrame.from_dict(Hierarchical_RI, orient='index')
        dfs["Hierarchical_RI"] = df_Hierarchical_RI.transpose()
        df_BIRCH_Purity = pd.DataFrame.from_dict(BIRCH_Purity, orient='index')
        dfs["BIRCH_Purity"] = df_BIRCH_Purity.transpose()
        df_Hierarchical_Purity = pd.DataFrame.from_dict(Hierarchical_Purity, orient='index')
        dfs["Hierarchical_Purity"] = df_Hierarchical_Purity.transpose()

        for key, value in dfs.items():
            print(key, value)
            value.to_csv("result/final_comparison/" + sub + "/" + key + sub + ".csv")


def star_pic(mode="Mix", choose='All'):
    data_path = "result/final_comparison/Mix/"
    files = [fn for fn in os.listdir(data_path) if fn.endswith(".csv")]
    dfs = {}
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    for file in files:
        dfs[file] = pd.read_csv(os.path.join(data_path, file), encoding="UTF-8", header=0, index_col=0)
        table = dfs[file]
        # print(table)
        labels = [Label(index) for index in table.index]
        if mode == "Mix":
            mix_drawing(file, table, colors)
        else:
            single_draw(file, mode, table, colors, labels, choose)


def Label(index):
    label = ''
    if "Method1" in index:
        label = "M1"
    else:
        if "subjectheader_" in index:
            label = "M6"
        if "subject_" in index:
            label = "M5"
        if "none_" in index:
            label = "M4"
    if "All" in index:
        label += "_Table"
    if "Subject_Col" in index:
        label += "_SC"
    if "shuffle_col,sample_row" in index:
        label += "_Aug1"
    if "shuffle_col,shuffle_row" in index:
        label += "_Aug2"
    return label


def choose_list(choice, table: pd.DataFrame):
    title = "All Methods On Whole Table"
    indexes = table.index.tolist()
    truncate_list = [indexes.index(i) for i in indexes]
    if "All_starmie" in choice:
        if "sample" in choice:
            truncate_list = [indexes.index(i) for i in indexes if
                             "shuffle_row" not in i and "All" in i]
            title =   "Sampling Rows and \n Shuffling Columns On Whole Table"
        if "shuffle_row" in choice:
            truncate_list = [indexes.index(i) for i in indexes if
                             "sample_row" not in i and "All" in i]
            title =  "Shuffling Rows and \n Shuffling Columns On Whole Table"
    if "Sub_starmie" in choice:
        if "sample" in choice:
            truncate_list = [indexes.index(i) for i in indexes if
                             "shuffle_row" not in i and "Subject_Col" in i]
            title =  "Sampling Rows and \n Shuffling Columns On Subject Column(s)"
        if "shuffle_row" in choice:
            truncate_list = [indexes.index(i) for i in indexes if
                             "sample_row" not in i and "Subject_Col" in i]
            title =  "Shuffling Rows and \n Shuffling Columns On Subject Column(s)"

    if "starmie_sample" in choice:
        truncate_list = [indexes.index(i) for i in indexes if
                         "sample_row" in i]
        title = "Sampling Rows and \n Shuffling Columns"
    if "starmie_shuffle" in choice:
        truncate_list = [indexes.index(i) for i in indexes if
                         "shuffle_row" in i]
        title = "Shuffling Rows and \n Shuffling Columns"
    if "WholeTable" in choice:
        truncate_list = [indexes.index(i) for i in indexes if
                         "All" in i and "Method1" not in i]
        title = "Different Augmentation \n Methods On Whole Table"
    if "SubCol" in choice:
        truncate_list = [indexes.index(i) for i in indexes if
                         "Subject_Col" in i and "Method1" not in i]
        title = "Different Augmentation \n Methods On Subject Column(s)"
    return truncate_list,title


def single_draw(file, dataset, table: pd.DataFrame, colors, labels, choose):
    truncate_list, title_components = choose_list(choose, table)
    print(choose, truncate_list)
    fig, ax = plt.subplots(figsize=(15, 13))
    width = 1
    fig.subplots_adjust(bottom=0.18)
    selected_index = [table.index[i] for i in truncate_list]
    table = table.loc[selected_index]
    colors = [colors[i] for i in truncate_list]
    labels = [labels[i] for i in truncate_list]
    x = np.arange(len(labels)) * 1.5
    for col_name, col_values in table.items():
        ax.bar(x, col_values, color=colors, width=width)
    ax.set_ylabel(file.split("_")[1][:-4], fontsize=18)
    ax.set_xlabel('Methods', fontsize=18)
    title = (" ").join(file[:-4].split("_")) \
            + " of "+title_components +" in " + dataset
    ax.set_title(title, fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    mkdir(os.getcwd() + "/fig/" + dataset + "/" + choose + "/")
    plt.savefig("fig/" + dataset + "/" + choose + "/" + file[:-4] + ".png")
    # plt.show()
    plt.close()


def mix_drawing(file, table, colors):
    x = np.arange(len(table.columns)) * 2
    width = 0.12
    fig, ax = plt.subplots(figsize=(15, 10))
    rects = []
    for i, (index, row) in enumerate(table.iterrows()):
        label = Label(index)
        if i != 0 or i != len(table) - 1:
            rect = ax.bar(x - width + 0.12 * i,
                          [round(num, 3) for num in row.tolist()], width, label=label, color=colors[i])

        else:
            rect = ax.bar(x - width * (len(table.columns) * 1) + 0.12 * i,
                          [round(num, 3) for num in row.tolist()], width, label=label, color=colors[i])
        rects.append(rect)

        # 为y轴、标题和x轴等添加一些文本。
    ax.set_ylabel(file.split("_")[1][:-4], fontsize=20)
    ax.set_xlabel('datasets', fontsize=20)
    ax.set_title((" ").join(file[:-4].split("_")) + " of All Methods", fontsize=30)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticks(x - 0.5)
    ax.set_xticklabels(table.columns)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=7, fontsize=12)

    def autolabel(rects):
        """label for the data"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

    # for rect in rects:
    #   autolabel(rect)
    fig.tight_layout()

    plt.savefig("fig/Mix/" + file[:-4] + ".png")
    plt.show()
    plt.close()
"""
"All", "All_starmie_sample",
           "All_starmie_shuffle_row",
           "Sub_starmie_sample",
           "Sub_starmie_shuffle_row",
           "starmie_sample", "starmie_shuffle"
"""
#
choices = [ "WholeTable", "SubCol"]
# dfs_sum(is_sub)
for i in DATA_PATH:
    for choice in choices:
        star_pic(mode=i, choose=choice)
