
"""df = pd.read_csv("datasets/WDC/Test/T2DV2_253.csv")
augment(df,"highlight_cells")
augment(df,"replace_high_cells")"""
import os

import pandas as pd

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
from starmie.sdd.augment import augment
table = pd.read_csv("datasets/WDC/Test/SOTAB_1.csv")
t2 = augment(table, "sample_cells_TFIDF")
t3 = augment(table, "replace_cells_TFIDF")

data_path = os.getcwd() + "/datasets/WDC/Test/"

files = [fn for fn in os.listdir(data_path)]

for file in files:
    print(file)
    table = pd.read_csv(os.path.join(data_path, file))
    t3 = augment(table, "replace_cells_TFIDF")
    t2 = augment(table, "sample_cells_TFIDF")