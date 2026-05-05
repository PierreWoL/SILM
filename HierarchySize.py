import networkx as nx
from collections import deque, defaultdict
import random
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd


# =========================================================
# Basic utilities
# =========================================================

def find_root(G: nx.DiGraph):
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got {roots}")
    return roots[0]


def ensure_tree_like_dag(G: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph must be a DAG.")
    root = find_root(G)
    bad = [n for n in G.nodes if n != root and G.in_degree(n) != 1]
    if bad:
        raise ValueError(
            "This implementation assumes a tree-like hierarchy "
            f"(each non-root node has exactly one parent). Bad nodes: {bad}"
        )


def compute_depths(G: nx.DiGraph, root):
    depths = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in G.successors(u):
            depths[v] = depths[u] + 1
            q.append(v)
    return depths


def get_parent(G: nx.DiGraph, node):
    preds = list(G.predecessors(node))
    return preds[0] if preds else None


def get_ancestor_chain_inclusive(G: nx.DiGraph, node):
    """
    Return [root, ..., node]
    """
    chain = [node]
    cur = node
    while True:
        p = get_parent(G, cur)
        if p is None:
            break
        chain.append(p)
        cur = p
    chain.reverse()
    return chain


def get_original_leaves(G: nx.DiGraph):
    return [n for n in G.nodes if G.out_degree(n) == 0]


def node_direct_support(G: nx.DiGraph, node):
    return len(G.nodes[node].get("tables", []))


def compute_cumulative_support(G: nx.DiGraph):
    """
    cumulative_support[n] = direct tables at n + all direct tables in descendants
    """
    support = {n: node_direct_support(G, n) for n in G.nodes}
    topo = list(nx.topological_sort(G))[::-1]
    for u in topo:
        for ch in G.successors(u):
            support[u] += support[ch]
    return support


def compute_root_branch_of_node(G: nx.DiGraph, root):
    """
    branch_of[node] = which direct child of root dominates this node
    root itself -> None
    """
    branch_of = {root: None}
    for ch in G.successors(root):
        stack = [ch]
        while stack:
            u = stack.pop()
            branch_of[u] = ch
            for v in G.successors(u):
                stack.append(v)
    print("direct child of root ->", branch_of)
    return branch_of


def closure_from_selected_leaves(G: nx.DiGraph, selected_leaves: Set):
    """
    Build ancestor closure of selected original leaves.
    """
    selected_nodes = set()
    for leaf in selected_leaves:
        selected_nodes.update(get_ancestor_chain_inclusive(G, leaf))
    return selected_nodes


def induced_subtree_from_leaves(G: nx.DiGraph, selected_leaves: Set):
    nodes = closure_from_selected_leaves(G, selected_leaves)
    return G.subgraph(nodes).copy()


def scenario_leaves(H: nx.DiGraph):
    return [n for n in H.nodes if H.out_degree(n) == 0]


def verify_original_leaf_preserving(H: nx.DiGraph, original_leaf_set: Set):
    bad = [n for n in scenario_leaves(H) if n not in original_leaf_set]
    if bad:
        raise ValueError(
            f"Scenario contains non-original leaves (pseudo-leaves): {bad}"
        )


def verify_connected_rooted_subtree(H: nx.DiGraph):
    root = find_root(H)
    if not nx.is_weakly_connected(H):
        raise ValueError("Scenario is not connected.")
    # tree-like check
    bad = [n for n in H.nodes if n != root and H.in_degree(n) != 1]
    if bad:
        raise ValueError(f"Scenario is not a rooted tree-like subtree. Bad nodes: {bad}")


# =========================================================
# Greedy leaf-selection logic
# =========================================================

def build_leaf_metadata(G: nx.DiGraph):
    """
    Precompute metadata for each original leaf.
    """
    ensure_tree_like_dag(G)
    root = find_root(G)
    depths = compute_depths(G, root)
    cumulative_support = compute_cumulative_support(G)
    branch_of = compute_root_branch_of_node(G, root)
    original_leaves = get_original_leaves(G)

    metadata = {}
    for leaf in original_leaves:
        chain = get_ancestor_chain_inclusive(G, leaf)
        metadata[leaf] = {
            "depth": depths[leaf],
            "branch": branch_of[leaf],
            "direct_support": node_direct_support(G, leaf),
            "cumulative_branch_support": cumulative_support[branch_of[leaf]] if branch_of[leaf] is not None else 0,
            "closure_nodes": set(chain),   # ancestor chain inclusive
        }
    return metadata


def score_candidate_leaf(
    leaf,
    metadata,
    current_leaves: Set,
    current_nodes: Set,
    target_num_nodes: int,
    covered_root_branches: Set,
    rng: random.Random,
    w_branch_gain=100.0,
    w_depth=10.0,
    w_support=1.0,
    w_branch_support=0.01,
    w_overshoot=5.0,
):
    """
    Higher is better.
    """
    info = metadata[leaf]
    new_nodes = info["closure_nodes"] - current_nodes
    new_total_nodes = len(current_nodes) + len(new_nodes)
    overshoot = max(0, new_total_nodes - target_num_nodes)

    branch = info["branch"]
    branch_gain = 1.0 if branch not in covered_root_branches else 0.0

    score = (
        w_branch_gain * branch_gain
        + w_depth * info["depth"]
        + w_support * info["direct_support"]
        + w_branch_support * info["cumulative_branch_support"]
        - w_overshoot * overshoot
        + 1e-6 * rng.random()
    )
    return score


def select_leaves_for_target(
    G: nx.DiGraph,
    target_num_nodes: int,
    min_leaf_support: int,
    min_root_branches: int,
    locked_leaves: Optional[Set] = None,
    seed: int = 42,
):
    """
    Greedy selection of original leaves.
    Returns:
        selected_leaves, selected_nodes, subtree
    """
    ensure_tree_like_dag(G)
    root = find_root(G)
    original_leaves = set(get_original_leaves(G))
    metadata = build_leaf_metadata(G)
    rng = random.Random(seed)

    # valid leaves
    valid_leaves = {
        leaf for leaf in original_leaves
        if metadata[leaf]["direct_support"] >= min_leaf_support
    }

    if not valid_leaves:
        raise ValueError("No valid original leaves satisfy min_leaf_support.")

    selected_leaves = set(locked_leaves) if locked_leaves else set()
    if not selected_leaves.issubset(valid_leaves):
        bad = selected_leaves - valid_leaves
        raise ValueError(f"Locked leaves not valid under min_leaf_support: {bad}")

    selected_nodes = closure_from_selected_leaves(G, selected_leaves)

    def current_branches():
        return {metadata[leaf]["branch"] for leaf in selected_leaves}

    # If nothing locked, seed with one strong leaf
    if not selected_leaves:
        candidates = list(valid_leaves)
        candidates.sort(
            key=lambda lf: (
                metadata[lf]["depth"],
                metadata[lf]["cumulative_branch_support"],
                metadata[lf]["direct_support"],
                str(lf),
            ),
            reverse=True
        )
        first_leaf = candidates[0]
        selected_leaves.add(first_leaf)
        selected_nodes = closure_from_selected_leaves(G, selected_leaves)

    # Greedily add leaves until:
    # 1) enough root branches covered
    # 2) enough total nodes
    while True:
        covered = current_branches()
        enough_branches = len(covered) >= min_root_branches
        enough_size = len(selected_nodes) >= target_num_nodes

        if enough_branches and enough_size:
            break

        remaining = [lf for lf in valid_leaves if lf not in selected_leaves]
        if not remaining:
            break

        remaining.sort(
            key=lambda lf: score_candidate_leaf(
                lf,
                metadata=metadata,
                current_leaves=selected_leaves,
                current_nodes=selected_nodes,
                target_num_nodes=target_num_nodes,
                covered_root_branches=covered,
                rng=rng,
            ),
            reverse=True
        )

        chosen = remaining[0]
        selected_leaves.add(chosen)
        selected_nodes = closure_from_selected_leaves(G, selected_leaves)

    H = G.subgraph(selected_nodes).copy()

    # checks
    verify_connected_rooted_subtree(H)
    verify_original_leaf_preserving(H, original_leaves)

    # Ensure root branch coverage
    final_branches = {metadata[leaf]["branch"] for leaf in selected_leaves}
    print(final_branches)
    if len(final_branches) < min_root_branches:
        raise ValueError(
            f"Could not satisfy min_root_branches={min_root_branches}. "
            f"Got only {len(final_branches)}."
        )

    return selected_leaves, selected_nodes, H


# =========================================================
# Nested scenario builder
# =========================================================

def build_nested_leaf_preserving_scenarios(
    G: nx.DiGraph,
    target_num_nodes: Dict[str, int],
    min_root_branches: Dict[str, int],
    min_leaf_support: int = 5,
    seed: int = 42,
):
    """
    Example:
        target_num_nodes = {
            "H25": 7,
            "H50": 12,
            "H75": 18,
            "H100": 24,
        }

        min_root_branches = {
            "H25": 2,
            "H50": 2,
            "H75": 3,
            "H100": 4,
        }
    """
    ensure_tree_like_dag(G)
    root = find_root(G)
    original_leaves = set(get_original_leaves(G))

    order = ["H25", "H50", "H75", "H100"]

    scenarios = {}
    selected_leaf_sets = {}

    locked_leaves = set()
    for i, name in enumerate(order):
        if name == "H100":
            H = G.copy()
            verify_connected_rooted_subtree(H)
            verify_original_leaf_preserving(H, original_leaves)

            # For consistency, recover H100 leaves from graph
            selected_leaves = set(scenario_leaves(H))
            selected_nodes = set(H.nodes)
        else:
            selected_leaves, selected_nodes, H = select_leaves_for_target(
                G=G,
                target_num_nodes=target_num_nodes[name],
                min_leaf_support=min_leaf_support,
                min_root_branches=min_root_branches[name],
                locked_leaves=locked_leaves,
                seed=seed + i,
            )

        scenarios[name] = H
        selected_leaf_sets[name] = selected_leaves
        locked_leaves = set(selected_leaves)  # nestedness at leaf level

    # nested sanity check
    for prev_name, next_name in [("H25", "H50"), ("H50", "H75"), ("H75", "H100")]:
        if not set(scenarios[prev_name].nodes).issubset(set(scenarios[next_name].nodes)):
            raise ValueError(f"Nestedness violated: {prev_name} is not subset of {next_name}")

    return scenarios, selected_leaf_sets


# =========================================================
# Diagnostics / reporting
# =========================================================

def summarize_scenario(G: nx.DiGraph, name: str):
    root = find_root(G)
    depths = compute_depths(G, root)
    levels = defaultdict(list)
    for n, d in depths.items():
        levels[d].append(n)

    leaves = scenario_leaves(G)
    total_direct_tables = sum(node_direct_support(G, n) for n in G.nodes)

    print(f"===== {name} =====")
    print(f"num_nodes = {G.number_of_nodes()}")
    print(f"num_edges = {G.number_of_edges()}")
    print(f"depth_profile = " + ", ".join(f"L{d}:{len(levels[d])}" for d in sorted(levels)))
    print(f"num_leaves = {len(leaves)}")
    print(f"total_direct_tables = {total_direct_tables}")
    print(f"leaves = {leaves}")
    print()


def print_root_branch_stats(G: nx.DiGraph):
    root = find_root(G)
    cumulative_support = compute_cumulative_support(G)

    for ch in G.successors(root):
        descendants = nx.descendants(G, ch)
        nodes = {ch} | descendants
        leaves = [n for n in nodes if G.out_degree(n) == 0]
        leaf_tables = sum(node_direct_support(G, lf) for lf in leaves)

        print(
            f"branch={ch:20s} "
            f"types={len(nodes):3d} "
            f"leaves={len(leaves):3d} "
            f"cum_support={cumulative_support[ch]:4d} "
            f"leaf_tables={leaf_tables:4d}"
        )


# =========================================================
# Optional: fixed table sampling per selected leaf
# =========================================================

def sample_tables_per_selected_leaf(H: nx.DiGraph, num_tables_per_leaf: int = 15, seed: int = 42):
    rng = random.Random(seed)
    leaf_samples = {}
    for leaf in scenario_leaves(H):
        tables = list(H.nodes[leaf].get("tables", []))
        if len(tables) < num_tables_per_leaf:
            raise ValueError(
                f"Leaf {leaf} has only {len(tables)} tables, cannot sample {num_tables_per_leaf}"
            )
        leaf_samples[leaf] = rng.sample(tables, num_tables_per_leaf)
    return leaf_samples







# ==========================================================
# Examples
# ==========================================================
import networkx as nx
import pickle
def get_sub_hierarchy(G, root_type):
    """
    Return the sub-hierarchy rooted at root_type.
    Assumption: edges are parent -> child.
    """
    if root_type not in G:
        raise ValueError(f"{root_type} is not in the graph.")

    descendants = nx.descendants(G, root_type)
    nodes = descendants | {root_type}

    subG = G.subgraph(nodes).copy()
    return subG


def normalize_tables(value):
    """
    Convert node table attribute to a clean list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, set):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def traverse_type_hierarchy_with_tables(G, root_type, table_attr="table"):
    """
    Traverse a type hierarchy from top to down.

    Assumption:
    - G is a networkx.DiGraph
    - Edge direction: parent_type -> child_type
    - Each node may have G.nodes[node][table_attr]
    """
    if root_type not in G:
        raise ValueError(f"{root_type} is not in the graph.")

    for node in nx.bfs_tree(G, root_type):
        depth = nx.shortest_path_length(G, root_type, node)
        tables = normalize_tables(G.nodes[node].get(table_attr, []))

        print("  " * depth + f"- {node}")
        print("  " * depth + f"  tables: {tables}")

file_path = "E:\Project\CurrentDataset\datasets\WDC\graphGroundTruth.pkl"
csv_path = "E:\Project\CurrentDataset\datasets\WDC\groundTruth.csv"
with open(file_path, "rb") as f:
    graph = pickle.load(f)
df = pd.read_csv(csv_path)

mapping_leaf = (
    df.groupby("class")["fileName"]
      .apply(list)
      .to_dict()
)

#print(mapping_leaf)
for type, tables in mapping_leaf.items():
    graph.nodes[type]["tables"] = tables
graph.remove_node("EmergencyService")
#graph.remove_edge("LocalBusiness", "Hospital")

place_hierarchy = get_sub_hierarchy(graph,"Place")
#traverse_type_hierarchy_with_tables(place_hierarchy, "Place",table_attr="tables" )
#parents = list(place_hierarchy.predecessors("Hospital"))
#print(parents)
# ... 继续加完整 H100

target_num_nodes = {
            "H25": 7,
            "H50": 12,
            "H75": 18,
            "H100": 24,
        }

min_root_branches = {
            "H25": 2,
            "H50": 2,
            "H75": 3,
            "H100": 4,
        }

scenarios, selected_leaf_sets = build_nested_leaf_preserving_scenarios(
    place_hierarchy,
    target_num_nodes=target_num_nodes,
    min_root_branches=min_root_branches,
    min_leaf_support=15,
    seed=42,
)

for name in ["H25", "H50", "H75", "H100"]:
    summarize_scenario(scenarios[name], name)

print_root_branch_stats(place_hierarchy)