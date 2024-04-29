
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
def compute_maxD(ancestors, graph,target_node):
    max_distance = -1
    NodeMax =None
    dict_dis = {}
    for an in ancestors:
        shortest_path_lengths = nx.shortest_path_length(graph.reverse(), target=an, source=target_node)
        print(f"节点 {an}: {shortest_path_lengths}")
        if shortest_path_lengths in dict_dis.keys():
            dict_dis[shortest_path_lengths].append(an)
        else:
            dict_dis[shortest_path_lengths] =[an]

    return dict_dis[max(list(dict_dis.keys()))]



# This is for drawing the paper example
ground_truth_csv = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "datasets/TabFact/Try.csv"))
#contains_sport = ground_truth_csv['LowestClass'].str.contains('sport', case=False, na=False)
contains_sport = ground_truth_csv["LowestClass"].str.contains("sport", case=False, na=False) & \
            (ground_truth_csv["LowestClass"].str.contains("competition", case=False, na=False) |
             ground_truth_csv["LowestClass"].str.contains("league", case=False, na=False) |
             ground_truth_csv["LowestClass"].str.contains("team", case=False, na=False))
result_df = ground_truth_csv[contains_sport]

print(result_df['LowestClass'].unique(), len(result_df['LowestClass'].unique()))

labels = os.listdir(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "datasets/TabFact/Label"))

G = nx.DiGraph()
len_G = 0

for index, row in result_df.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:7]
        all_nodes = set(df.values.ravel())
        if len(set(G.nodes())):
            all_nodes = all_nodes - set(G.nodes())
        else:
            nodes_set = all_nodes
        G.add_nodes_from(all_nodes)
        nodes_set = set(G.nodes())
        for _, row2 in df.iterrows():
            labels_table = row2.dropna().tolist()
            for i in range(len(labels_table) - 1):
                if labels_table[i + 1]!=labels_table[i]:
                    G.add_edge(labels_table[i + 1], labels_table[i])

    else:
        if row["class"]!=" ":
            superclass = row["superclass"]
            classX = row["class"]
            all_nodes = {superclass, classX}
            all_nodes = all_nodes - set(G.nodes())
            if len(all_nodes) > 0:

                G.add_nodes_from(all_nodes)
                G.add_edge(superclass, classX)
    if len_G<len(set(G.nodes())):
        graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        plt.figure(figsize=(40, 32))
        nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue",   font_size=11,arrowsize=20)
        plt.show()
        len_G = len(set(G.nodes()))


