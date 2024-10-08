import os

import pandas as pd
import matplotlib.pyplot as plt

colM = [ "RoBERTa_Instance", "SBERT_Instance","RoBERTa_FineTune","EmbDI", "distribution_based","RoBERTa_Schema", "SBERT_Schema",   "cupid", "similarity_flooding"]#"RoBERTa_FineTune",
box_colorsM = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray', 'lightcyan','blue','green','purple']
col1M = [ "RoBERTa_Instance", "SBERT_Instance","RoBERTa_FineTune","EmbDI", "distribution_based"]#"RoBERTa_FineTune",
box_colors1M = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
col2M = ["RoBERTa_Schema", "SBERT_Schema",   "cupid", "similarity_flooding"]
box_colors2M = ['lightcyan','blue','green','purple']
#data = pd.read_excel('D:\CurrentDataset\ValentineDatasets\TPC-DI\View-Unionable\View-Unionable.xlsx', sheet_name=1)
def draw(box_colors,columns,y_label, store_path,data_path, title,file_name):

    if not os.path.exists(data_path):
        print(f"file '{data_path}' not exist, check the suffix!")
        return False
    else:
        if data_path.endswith(".xlsx"):
            data = pd.read_excel(data_path, sheet_name=file_name)
        else:
            data = pd.read_csv(data_path, encoding='latin1')

        data_to_plot = data[columns]
        print(data_to_plot)
        plt.figure(figsize=(10, 8))
        bp = plt.boxplot(data_to_plot.values, labels=data_to_plot.columns, patch_artist=True)
        for box, color in zip(bp['boxes'], box_colors):
            box.set(facecolor=color)
        plt.xlabel('Embedding methods', wrap=True, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        # plt.tight_layout()
        plt.xticks(rotation=15, fontsize=12)

        plt.title(title, fontsize=19)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(top=0.9, bottom=0.12)
        plt.legend()
        """medians = [median.get_ydata()[0] for median in bp['medians']]
        means = [mean.get_ydata()[0] for mean in bp['means']]
        positions = range(len(columns))
        for pos, median, mean in zip(positions, medians, means):
            plt.text(pos + 1, median + 0.02, f"Median: {median:.3f}", ha='center', va='bottom', fontsize=8)
            plt.text(pos + 1, mean - 0.02, f"Mean: {mean:.3f}", ha='center', va='top', fontsize=8)"""

        medians = [median.get_ydata()[0] for median in bp['medians']]
        means = [mean.get_ydata()[0] for mean in bp['means']]
        xtick_labels = data_to_plot.columns
        for i, median, mean in zip(range(len(xtick_labels)), medians, means):
            plt.text(i + 1 - 0.25, median, f'Median: {median:.3f}', ha='right', va='center', color='red', fontsize=8)
            plt.text(i + 1 - 0.25, mean, f'Mean: {mean:.3f}', ha='right', va='center', color='blue', fontsize=8)
        plt.savefig(store_path)
        plt.show()


"""datasets = ["Magellan", "ChEMBL","TPC-DI", "OpenData"]  # "ChEMBL",, "Magellan" 'Wikidata'
types = ['Joinable','Semantically-Joinable','Unionable','View-Unionable']
for dataset in datasets:
    parent_path = os.path.join(os.getcwd(), "../ValentineDatasets", dataset)
    if dataset != 'wikidata' and dataset!='Magellan':
        for type in types:
            xlsx_sum_p = os.path.join(parent_path,type,type+".xlsx")
            store_p= os.path.join(parent_path,type)
            try:
                draw(box_colorsM,colM,os.path.join(store_p,type+"1.pdf"), xlsx_sum_p,dataset,type,1)
                draw(box_colors1M,col1M,os.path.join(store_p,type+"2.pdf"), xlsx_sum_p, dataset, type,2)
                draw(box_colors2M,col2M,os.path.join(store_p,type+"3.pdf"), xlsx_sum_p, dataset, type,3)
            except FileNotFoundError as e:
                print(e)
    else:
        xlsx_sum_p = os.path.join(parent_path, dataset + ".xlsx")
        store_p = os.path.join(parent_path, dataset + ".png")
        try:
            draw(box_colorsM,colM,store_p, xlsx_sum_p,dataset,"Unionable",1)
            draw(box_colors1M,col1M,store_p, xlsx_sum_p, dataset, "Unionable",2)
            draw(box_colors2M,col2M,store_p, xlsx_sum_p, dataset, "Unionable",3)
        except FileNotFoundError as e:
            print(e)
"""

