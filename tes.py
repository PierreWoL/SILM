import ast
import os
import pickle

import networkx as nx
import pandas as pd

"""
# 创建一个有向图并添加节点和边
G = nx.DiGraph()
G.add_edge("A", "B")
G.add_edge("A", "C")
G.add_edge("B", "D")


# 要检查的节点
node_A = "A"

# 获取节点 A 的所有出节点（直接后继节点）
successors_list = list(G.successors(node_A))
te = nx.descendants(G, node_A)
asn = nx.ancestors(G, node_A)
print(f"{node_A} 的出节点列表：{successors_list}")
print(f"{node_A} 祖先列表：{asn}")
print(f"{node_A} 不知道什么玩意：{te}")
dataset = "WDC"
path = 'result/embedding/starmie/vectors/%s' % dataset
files = [file for file in os.listdir(path) if "Pretrain" not in file and file.endswith("pkl")]
for file in files:
    with open(os.path.join(path, file), "rb") as f:
        G_tree = pickle.load(f)
        print(file, G_tree[200][0], len(G_tree[200][1]))



target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
with open(os.path.join(target_path, "TypeDict.pkl"), "rb") as file:
    new_dict = pickle.load(file)

inst = pd.read_csv(os.path.join(os.getcwd(), "datasets/TabFact/New.csv"))
for index, row in inst.iterrows():
    #print(index,row,row["Lowest level type"],new_dict[row["Lowest level type"]],inst.iloc[index,2])
    inst.iloc[index,2] = str(new_dict[row["Lowest level type"]])
inst.to_csv(os.path.join(os.getcwd(), "datasets/TabFact/New.csv"),index=False)
"""
"""import networkx as nx
import json
excpetion = []
# 读取JSON文件
data=pd.read_csv("schemaorg-all-http-types.csv")
prefix = "http://schema.org/"

# 创建一个空的DiGraph
G = nx.DiGraph()
# 遍历JSON数据的"@graph"部分
for index,row in data.iterrows():
    # 获取节点的label和@id
    node_id = row["label"]
    # 添加节点到图中
    G.add_node(node_id, label=node_id)
    if pd.isna(row["subTypeOf"]) is False:
        if "," in row["subTypeOf"]:
            parent_label = row["subTypeOf"].split(", ")
            print(parent_label)
            for i in parent_label:
                print(i[len(prefix):], "is parent label of ", node_id)
                G.add_edge(i[len(prefix):], node_id)
        else:
            parent_label = row["subTypeOf"][len(prefix):]
            G.add_edge(parent_label, node_id)

unimportant = ['Thing', 'Boolean', 'Text', '2000/01/rdf-schema#Class', 'Date', 'DateTime', 'False', 'Number',
               'Time', 'True','DataType','Float', 'Integer', 'URL','XPathType',
               'PronounceableText', 'CssSelectorType','StupidType']
for i in unimportant:
    G.remove_node(i)


top_level_typ = [i for i in G.nodes() if G.in_degree(i) == 0]
print(top_level_typ,len(G.nodes()))

with open(os.path.join( "schemaorgTree.pkl"), "wb") as file:
    pickle.dump(G, file)

"""

"""
path = "datasets/TabFact/groundTruth.csv"
df = pd.read_csv(path).dropna()
df = df[df['superclass'].str.contains('\[')]
df['superclass'] = df['superclass'].apply(ast.literal_eval)

# 展开superclass列
df = df.explode('superclass')

# 对展开后的数据进行分组并统计
summary = pd.DataFrame(df.groupby('superclass').size().reset_index(name='count'))
summary.to_csv("datasets/TabFact/agre1.csv")
print(summary)

similar_item = [('AcademicJournal', 'Manuscript'),
                ('AdministrativeRegion', 'DefinedRegion'),
                ('BaseballPlayer', 'Person'),
                ('Bird', 'Animal'),
                ('Mammal', 'Animal'),
                ('Building', 'LandmarksOrHistoricalBuildings'),
                ('Company', 'LocalBusiness'),
                ('cricketer', 'Person'),
                ('Currency', 'Property'),
                ('FictionalCharacter', 'Person'),
                ('GolfPlayer', 'Person'), ('Monarch', 'Person'), ('Wrestler', 'Person'),
                ('Saint', 'Person'), ('Scientist', 'Person'), ('swimmer', 'Person'),
                ('Novel', 'Manuscript'), ('Plant','Organism'),
                ('Election', 'PoliticalParty'),
                ('TelevisionShow', 'CreativeWorkSeries')]

with open("schemaorgTree.pkl", "rb") as file:
    G = pickle.load(file)
for tuple_type in similar_item:
    G.add_edge(tuple_type[1], tuple_type[0])

top = [i for i in G.nodes()  if G.in_degree(i) == 0]
print(top)
"""
"""
wdc_label = pd.read_csv("datasets/WDC/groundTruth.csv")

for index, row in wdc_label.iterrows():
    lowest_type = row["Label"]
    if lowest_type in G.nodes():
        top_ancestor = [i for i in nx.ancestors(G, lowest_type) if G.in_degree(i) == 0]
        # print(lowest_type,top_ancestor)
    else:
        if row["Label"] == 'Political Party':
            lowest_type = "PoliticalParty"
        if row["Label"] == 'Film':
            lowest_type = "Movie"
        if row["Label"] == 'Lake':
            lowest_type = "LakeBodyOfWater"
        if row["Label"] == 'mountain':
            lowest_type = "Mountain"

        if row["Label"] == 'University':
            lowest_type = "CollegeOrUniversity"
        wdc_label.iloc[index,1] = lowest_type
        top_ancestor = [i for i in nx.ancestors(G, lowest_type) if G.in_degree(i) == 0]
    wdc_label.iloc[index, 2] = str(top_ancestor) if len(top_ancestor)>0 else [lowest_type]

wdc_label.to_csv("datasets/WDC/groundTruth.csv",index=False)
with open(os.path.join( "datasets/WDC/schemaorgTree.pkl"), "wb") as file:
    pickle.dump(G, file)"""

from collections import deque
import ast
import os
import pickle
import os.path
import pickle
import sys
import io
import requests

from concurrent.futures import ThreadPoolExecutor




target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
def run_hierarchy():
    ground_label_name1 = "01SourceTables.csv"
    data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
    ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
    result_dict = dict(zip(ground_truth_csv.iloc[:, 0], ground_truth_csv.iloc[:, 2]))
    names = ground_truth_csv["fileName"].unique()
    labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
    no_labels = [i for i in names if i not in labels]
    # ground_truth_csv = ground_truth_csv[ground_truth_csv["fileName"].isin(no_labels)]
    ground_truth = dict(zip(ground_truth_csv.iloc[:, 0], ground_truth_csv.iloc[:, 4]))

    similar_words = {}
    with open("filter_sim_all.pkl", "rb") as file:
        all_sims = pickle.load(file)
    for key, value in all_sims.items():
        for tuple in value.keys():
            word = tuple[0]
            if tuple[0] in similar_words.keys():
                if tuple[1] not in similar_words[word]:
                    similar_words[word].append(tuple[1])
            else:
                similar_words[word] = [tuple[1]]

    for word, similar_word_list in similar_words.items():
        if len(similar_word_list) == 1:
            similar_words[word] = similar_word_list[0]
    print(similar_words)
    # unique_items = list(set(similar_words.values()))

    node_length = 0
    G = nx.DiGraph()
    for index, row in ground_truth_csv.iterrows():
        if row["fileName"] in labels:
            label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
            df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
            for _, row2 in df.iterrows():
                labels_table = row2.dropna().tolist()
                for i in range(len(labels_table) - 1):
                    if labels_table[i + 1] != labels_table[i]:
                        # if labels_table[i + 1] not in abstract and labels_table[i] not in abstract:
                        child_type = labels_table[i]
                        if labels_table[i + 1] in G.nodes():
                            if labels_table[i] not in nx.ancestors(G, labels_table[i + 1]):
                                G.add_edge(labels_table[i + 1], child_type)
                                continue
                        else:
                            G.add_edge(labels_table[i + 1], child_type)
                            continue
        else:
            if row["class"] != " ":
                superclass = row["class"]
                classX = row["superclass"]
                all_nodes = {superclass, classX}
                all_nodes = all_nodes - set(G.nodes())
                G.add_nodes_from(all_nodes)
                G.add_edge(superclass, classX)

    with open(os.path.join(target_path, "graphGroundTruth2.pkl"), "wb") as file:
        pickle.dump(G, file)

file_path = os.path.join(os.path.join(target_path, "graphGroundTruth3.pkl"))
print(file_path)
with open(file_path, "rb") as file:
    G = pickle.load(file)
    Top_level_nodes = [i for i in G.nodes if G.in_degree(i) == 0]
print(Top_level_nodes,len(Top_level_nodes))

labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
ground_label_name = "01SourceTables.csv"
data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name)
ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
for index, row in ground_truth_csv.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
        lowest_types = df.iloc[:, 0].unique()
        top_level_types = []
        for type_low in lowest_types:
            if type_low in G.nodes():
                parent_top_per = [item for item in nx.ancestors(G, type_low) if G.in_degree(item) == 0]
                print(type_low,parent_top_per)
                for top_per in parent_top_per:
                    if top_per not in top_level_types:
                        top_level_types.append(top_per)
        ground_truth_csv.iloc[index, 4] = lowest_types
        ground_truth_csv.iloc[index, 5] = top_level_types
ground_truth_csv.to_csv(os.path.join(target_path, "new_test_origin3.csv"))