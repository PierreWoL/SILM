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

