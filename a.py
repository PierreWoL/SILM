"""df = pd.read_csv("datasets/WDC/Test/T2DV2_253.csv")
augment(df,"highlight_cells")
augment(df,"replace_high_cells")"""
import os

import pandas as pd
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
print(f"{node_A} 的出节点列表：{successors_list}")
print(f"{node_A} 祖先列表：{as_root}")
print(f"{node_A} 不知道什么玩意：{te}")"""

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
import numpy as np
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

# 将相似度转换为距离
distances = {k: 1 - v for k, v in similarities.items()}

# 创建一个距离矩阵
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

# 设置标题和轴标签的字体大小

plt.xlabel('Tables', fontsize=15)
plt.ylabel('Distance', fontsize=15)

# 设置y轴的刻度和字体大小
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.savefig("dendrogramExample.pdf")
plt.savefig("dendrogramExample.png")
plt.show()
