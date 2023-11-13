"""df = pd.read_csv("datasets/WDC/Test/T2DV2_253.csv")
augment(df,"highlight_cells")
augment(df,"replace_high_cells")"""
import concurrent
import os

import pandas as pd
import pickle
from plotly.figure_factory._dendrogram import sch

"""G = nx.DiGraph()
G.add_edge("A", "B")
G.add_edge("A", "C")
G.add_edge("B", "D")


# 要检查的节点
node_A = "D"

# 获取节点 A 的所有出节点（直接后继节点）
successors_list = list(G.successors(node_A))
te = nx.descendants(G, node_A)
asn = nx.ancestors(G, node_A)
as_root  = [item for item in asn if G.in_degree(item) == 0]
print(f"{node_A} out nodes：{successors_list}")
print(f"{node_A} ancestors：{as_root}")
print(f"{node_A} WTF：{te}")"""

"""from starmie.sdd.augment import augment
table = pd.read_csv("datasets/WDC/Test/T2DV2_7.csv")
t2 = augment(table, "sample_row_TFIDF")


data_path = os.getcwd() + "/datasets/WDC/Test/"
from d3l.utils.functions import unpickle_python_object
data_path2 = os.getcwd() + "/datasets/WDC/graphGroundTruth.pkl"
schema_org = unpickle_python_object(data_path2)
as_root  = [item for item in schema_org.nodes() if schema_org.in_degree(item) == 0]
print(as_root)"""
"""files = [fn for fn in os.listdir(data_path)]
#'replace_cells_TFIDF' ,
augmentation = ['sample_row_TFIDF', 'sample_cells_TFIDF']
for aug in augmentation:
    for file in files[300:]:

        table = pd.read_csv(os.path.join(data_path, file))
        print(file, aug)
        t2 = augment(table, aug, isTabFact=False)
        # print(table.transpose(),"\n", t2.transpose())
        break"""

"""
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# 数据
data = np.array([
    [0,1,0,0],
    [0,1.1,0,0],
    [0,1.2,0,0],
    [0,1.3,0,0],
    [0,1.4,0,0],
    [0,1,2,0],
    [0,1.1,2,0],
    [0,1.2,2,0]
])

print(data)
tsne = TSNE(n_components=2, perplexity=3, random_state=0)
transformed_data = tsne.fit_transform(data)

df = pd.DataFrame(transformed_data, columns=['Component 1', 'Component 2'])
df['label'] = ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6", "data_7", "data_8"]
df['cluster'] = ['Cluster 1', 'Cluster 1', 'Cluster 1', 'Cluster 1', 'Cluster 1', 'Cluster 2', 'Cluster 2', 'Cluster 2']
print(df)
# 使用 plotly 进行绘图
fig = px.scatter(df, x='Component 1', y='Component 2', text='label', hover_data=['label'], color='cluster')
fig.update_traces(marker=dict(size=12),
                  selector=dict(mode='markers+text'))
fig.show()
fig.write_html("output_plot1.html")
    
    
    
"""

"""import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 你的相似度数据
similarities = {
    'sim(1,2)': 0.75,
    'sim(1,3)': 0.2,
    'sim(1,4)': 0.33,
    'sim(2,3)': 0.2,
    'sim(2,4)': 0.17,
    'sim(3,4)': 0.75
}

 
distances = {k: 1 - v for k, v in similarities.items()}
 
dist_matrix = np.array([
    [0, distances['sim(1,2)'], distances['sim(1,3)'], distances['sim(1,4)']],
    [0, 0, distances['sim(2,3)'], distances['sim(2,4)']],
    [0, 0, 0, distances['sim(3,4)']],
    [0, 0, 0, 0]
])

# 转换成压缩向量形式
dist_array = dist_matrix[np.triu_indices(4, k = 1)]

# 进行层次聚类
Z = linkage(dist_array, 'single')

# 画出树状图
plt.figure(figsize=(6, 6))
dendrogram(Z, labels=['t1', 't2', 't3', 't4'])

 

plt.xlabel('Tables', fontsize=15)
plt.ylabel('Distance', fontsize=15)

 
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.savefig("dendrogramExample.pdf")
plt.savefig("dendrogramExample.png")
plt.show()
"""

"""from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
X = [[i] for i in [2, 8, 4, 1, 9, 9,12,11,3.7,6,7.7,8,5.6,17,5,7]]
print(len(X))
Z = linkage(X, 'ward')
print(Z)
fig = plt.figure(figsize=(5, 5))
dn = dendrogram(Z)
Z = linkage(X, 'single')
fig = plt.figure(figsize=(5, 5))
dn = dendrogram(Z)
plt.show()"""
"""target_path = os.path.join(os.getcwd(), "datasets/TabFact" )
with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
    G = pickle.load(file)

nodes = [i for i in G.nodes() if G.in_degree(i) == 0]
print(nodes)
subEventNodes = [i for i in G.successors('Event')]
subCreativeworkNodes = [i for i in G.successors('CreativeWork')]
print(subEventNodes)
print(subCreativeworkNodes)

G.remove_node("Event")
G.remove_node("CreativeWork")
Tops = [i for i in G.nodes() if G.in_degree(i)==0]
print(len(Tops), Tops)

with open(os.path.join(target_path, "graphGroundTruth01.pkl"), "wb") as file:
  pickle.dump(G, file)"""

"""df = pd.read_csv('datasets/TabFact/groundTruth.csv')
multi = pd.read_csv('datasets/TabFact/column_gt.csv',encoding='latin1')
for index, row in df.iterrows():
    tableName = row['fileName']
    label = row['superclass']
    lowLabel = row['class']

    mask = multi['fileName'] == tableName
    multi.loc[mask, 'LowestClass'] = lowLabel
    multi.loc[mask, 'TopClass'] = label

multi.to_csv('datasets/TabFact/column_gt.csv', index=False)"""


"""all = pd.read_excel("D:/CurrentDataset/datasets/TabFact/relationships.xlsx", sheet_name="Sheet1")
tables = all["TableName"].unique()
lowest_type = all["LowType"].unique()
column = all["ColumnLabel"].unique()"""

from SPARQLWrapper import SPARQLWrapper, JSON

all_dict = {}


def get_wikidata_id_from_wikipedia_url(row):
    url = row["url"]
    table = row["fileName"]
    # Extract the title from the Wikipedia URL
    title = url.split('/')[-1].replace('_', ' ')

    # Initialize the SPARQL query
    sparql_query = f"""SELECT ?item WHERE {{
  ?article schema:about ?item .
  ?article schema:isPartOf <https://en.wikipedia.org/> .
  ?article schema:name "{title}"@en .  
}}"""

    # Set up the SPARQL wrapper to query the Wikidata endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    # Execute the query and return the results
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        all_dict[table] = result["item"]["value"].split("/")[-1]


# Test the function with the given URL
# url = "https://en.wikipedia.org/wiki/1963_World_Wrestling_Championships"
# wikidata_id = get_wikidata_id_from_wikipedia_url(url)
# print(wikidata_id)
"""

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    re = executor.map(get_wikidata_id_from_wikipedia_url, rows)"""
# pickle_python_object(all_dict, "D:/CurrentDataset/datasets/TabFact/ids.pkl")


from SPARQLWrapper import SPARQLWrapper, JSON


def query_wikidata_relationship(dicts, entity1, entity2):
    # SPARQL query to find the relationship between two entities
    sparql_query = """
    SELECT ?property ?propertyLabel WHERE {{
      wd:{} ?property wd:{} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """.format(entity1, entity2)

    # Set up the SPARQL wrapper to query the Wikidata endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    # Execute the query and return the results
    results = sparql.query().convert()
    relationships = []
    for result in results["results"]["bindings"]:
        relationships.append(result["propertyLabel"]["value"].split("/")[-1])
    if len(relationships)>0:
        print((entity1,entity2),relationships )
        dicts[(entity1,entity2)] = relationships
from d3l.utils.functions import pickle_python_object, unpickle_python_object
ori = pd.read_csv("datasets/TabFact/Try.csv", encoding="latin1")
alls = unpickle_python_object("datasets/TabFact/ids.pkl")
ori = ori[ori["fileName"].isin(list(alls.keys()))]
#print(alls, len(alls))
# Forrest Gump (Q134773) and Tom Hanks (Q2263)
#relationships = query_wikidata_relationship("Q134773", "Q2263")
#print(relationships)

tables = ori[ori["fileName"].isin(list(alls.keys()))]
lowclass = list(tables["class"].unique())

top_class = list(tables["superclass"].unique())
# tables.to_csv("D:/CurrentDataset/datasets/TabFact/keeps.csv")
filtered_df_team = tables[tables['class'].str.contains('team|league|Team|club', regex=True, case=False) & ~tables['class'].str.contains('season',
                                                                                                        regex=True, case=False)]
print(len(filtered_df_team))
team_ids = {key: value for key, value in alls.items() if key in filtered_df_team["fileName"].unique()}
filtered_df_season = tables[tables['superclass'].str.contains('competition', regex=True, case=False)]
season_ids = {key: value for key, value in alls.items() if key in filtered_df_season["fileName"].unique()}

relationship_dict = {}
from concurrent.futures import ThreadPoolExecutor
with  ThreadPoolExecutor(max_workers=1000) as executor:
    future_to_relationship = {executor.submit(query_wikidata_relationship,relationship_dict, team_id, season_id): (team_id, season_id)
                              for team_table, team_id in team_ids.items()
                              for season_table, season_id in season_ids.items()}

print(relationship_dict)
pickle_python_object(relationship_dict, "datasets/TabFact/relationshipTeamSeason.pkl")