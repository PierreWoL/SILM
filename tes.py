import os
import pickle

import networkx as nx

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
dataset = "TabFact"
path = 'result/embedding/starmie/vectors/%s' % dataset
files = [file for file in os.listdir(path) if "Pretrain" in file and file.endswith("pkl")]
for file in files:
    with open(os.path.join(path, file), "rb") as f:
        G_tree = pickle.load(f)
        print(file, G_tree[200][0], len(G_tree[200][1]))
