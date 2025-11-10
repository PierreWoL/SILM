import networkx as nx
import pickle
from collections import defaultdict


def bottom_up_levels_from_top(G: nx.DiGraph(), top_node):
    leaves = [n for n in nx.descendants(G, top_node) if G.out_degree(n) == 0]
    print(f"叶子节点: {leaves}")
    visited = set()
    current_level = set(leaves)
    level_num = 1
    current_level = set()
    for leaf in leaves:
        for parent in G.predecessors(leaf):
            current_level.add(parent)

    while current_level:
        print(f"\nLevel {level_num}（距离底部 {level_num} 层）: {current_level}")
        for node in current_level:
            children = list(G.successors(node))
            print(f"\nThe {node} has children: {children}")
        visited.update(current_level)

        next_level = set()
        for node in current_level:
            for parent in G.predecessors(node):
                if parent not in visited:
                    next_level.add(parent)

        current_level = next_level
        level_num += 1

        if top_node in current_level:
            print(f"\nLevel {level_num}（Top-Level Type: {top_node}）: {current_level}")
            break


def build_complete_bottom_up_levels(G, top_node):
    # 所有叶子作为第0层
    leaves = {n for n in nx.descendants(G, top_node) if G.out_degree(n) == 0}
    levels = [leaves]
    processed = set(leaves)

    while True:
        current_level = set()

        # 候选节点：还没处理的那些 descendants
        candidates = [n for n in nx.descendants(G, top_node) if n not in processed]

        for node in candidates:
            children = set(G.successors(node))
            # 如果所有 children 都已经被处理（不要求同一层），就可以处理当前 node
            if children.issubset(processed):
                current_level.add(node)

        if not current_level:
            break

        levels.append(current_level)
        processed.update(current_level)

    return levels  # 最底层在前，最顶层在后


# list = ['Place', 'Intangible', 'Organization', 'CreativeWork', 'Event', 'Person', 'Animal', 'Organism']

def check_levels_coverage(G, top_node, levels):
    # 目标子图：top_node 的所有后代（不包括 top_node 本人）
    target_nodes = set(nx.descendants(G, top_node))

    # 实际覆盖到的节点
    covered_nodes = set()
    for level in levels:
        for node in level:
            if node in covered_nodes:
                print(f"⚠️ 警告：节点 '{node}' 出现在多个层中")
            covered_nodes.add(node)

    # 检查遗漏
    missed = target_nodes - covered_nodes
    extra = covered_nodes - target_nodes

    if not missed and not extra:
        print("✅ 检查通过：没有遗漏也没有多余节点\n")
    else:
        if missed:
            print(f"❌ 漏掉了 {len(missed)} 个节点：\n{missed}\n")
        if extra:
            print(f"❌ 多了 {len(extra)} 个不属于子图的节点：\n{extra}\n")


# bottom_up_levels_from_top(G, 'Place')
# with open(r'datasets\WDC\graphGroundTruth.pkl', 'rb') as f:
# G = pickle.load(f)
"""

top_level_types  = list(G.successors("Thing"))
print("Top-Level Types:", top_level_types)
for top_node in top_level_types:
    levels = build_complete_bottom_up_levels(G, top_node=top_node)
    print("🧭 从最底层往上逐层打印：\n")
    for i, level in enumerate(levels):
        print(f"Level{i}（距离底部第 {i} 层，共 {len(levels)} 层）:")
        for node in sorted(level):
            print(f"  - {node}")
        if i != 0:
            for node in level:
                children = list(G.successors(node))
                print(f"The {node} has children: {children}")
    check_levels_coverage(G, top_node, levels)
"""
# bottom_up_levels_from_top(G, 'Place')
# levels = build_strict_bottom_up_levels(G, top_node='Event')
