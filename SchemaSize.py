import os
import ast
import pickle
import shutil
from pathlib import Path

import pandas as pd
import networkx as nx


def parse_superclass(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return v
        return [v]
    except Exception:
        return [str(x)]


def load_schema(schema_path):
    schema_path = Path(schema_path)

    if schema_path.suffix in [".pkl", ".pickle"]:
        with open(schema_path, "rb") as f:
            G = pickle.load(f)

    elif schema_path.suffix == ".graphml":
        G = nx.read_graphml(schema_path)

    elif schema_path.suffix == ".gml":
        G = nx.read_gml(schema_path)

    elif schema_path.suffix == ".csv":
        # Assume schema CSV has columns: parent, child
        df = pd.read_csv(schema_path)
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row["parent"], row["child"])

    else:
        raise ValueError(f"Unsupported schema format: {schema_path.suffix}")

    return G


def sample_tables(mapping_df, ratio, seed):
    """
    Randomly sample a given ratio of tables.
    """
    if ratio == 1.0:
        return mapping_df.copy()

    n_total = len(mapping_df)
    n_sample = max(1, round(n_total * ratio))

    return mapping_df.sample(n=n_sample, random_state=seed).copy()


def restrict_schema_to_sampled_tables(G, sampled_mapping_df):
    """
    Keep sampled tables' most specific types and their ancestors.

    Assumption:
    - G edges are parent -> child
    - sampled_mapping_df has columns: fileName, class, superclass
    """
    selected_types = set(sampled_mapping_df["class"].dropna())

    keep_nodes = set()

    for t in selected_types:
        if t not in G:
            print(f"[Warning] Type {t} not found in schema.")
            continue

        keep_nodes.add(t)
        keep_nodes.update(nx.ancestors(G, t))

    restricted_G = G.subgraph(keep_nodes).copy()

    # Reset table attributes
    for node in restricted_G.nodes:
        restricted_G.nodes[node]["tables"] = []

    # Attach sampled tables to their most specific types
    for _, row in sampled_mapping_df.iterrows():
        file_name = row["fileName"]
        type_name = row["class"]

        if type_name in restricted_G:
            restricted_G.nodes[type_name]["tables"].append(file_name)

    return restricted_G


def create_schema_size_subsets(
    mapping_csv,
    column_mapping_csv,
    schema_path,
    ori_tab_path,
    output_dir,
    ratios=(0.25, 0.50, 0.75, 1.0),
    num_repeats=3,
    base_seed=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_df = pd.read_csv(mapping_csv)
    column_mapping_df = pd.read_csv(column_mapping_csv)
    mapping_df["superclass_parsed"] = mapping_df["superclass"].apply(parse_superclass)

    G = load_schema(schema_path)

    print(f"Original number of tables: {len(mapping_df)}")
    print(f"Original number of column GT rows: {len(column_mapping_df)}")
    print(f"Original number of schema types: {G.number_of_nodes()}")
    print(f"Original number of schema edges: {G.number_of_edges()}")

    all_summaries = []

    for ratio in ratios:
        ratio_name = f"{int(ratio * 100)}"

        # For 100%, one run is usually enough if the input is identical.
        current_repeats = 1 if ratio == 1.0 else num_repeats

        for repeat_id in range(current_repeats):
            seed = base_seed + repeat_id

            sampled_df = sample_tables(mapping_df, ratio, seed)
            sampled_file_names = set(sampled_df["fileName"].tolist())

            subset_dir = output_dir / f"tables_{ratio_name}pct" / f"seed_{seed}"
            store_dir = subset_dir / "Test/"
            store_dir.mkdir(parents=True, exist_ok=True)
            ## 1. Save sampled tables
            sample_files = sampled_df["fileName"].tolist()

            source_folder = Path(ori_tab_path)
            target_folder = Path(store_dir)
            for filename in sample_files:
                src = source_folder / filename
                dst = target_folder / filename

                if src.exists() and src.is_file():
                    shutil.copy2(src, dst)  # copy2 会保留修改时间等元信息
                    print(f"Copied: {filename}")
                else:
                    print(f"Not found: {filename}")
            # sampled_table_list_path = subset_dir / "sampled_table_list.csv"
            #sampled_df[["fileName"]].to_csv(sampled_table_list_path, index=False)


            # 2. Save sampled mapping
            sampled_mapping_path = subset_dir / "groundTruth.csv"
            sampled_df[["fileName", "class", "superclass"]].to_csv(
                sampled_mapping_path,
                index=False
            )

            # 3. Save sampled column ground truth
            sampled_column_gt_df = column_mapping_df[
                column_mapping_df["fileName"].isin(sampled_file_names)
            ].copy()

            sampled_column_gt_path = subset_dir / "column_gt.csv"
            sampled_column_gt_df.to_csv(sampled_column_gt_path, index=False)

            # 4. Build restricted schema
            restricted_G = restrict_schema_to_sampled_tables(G, sampled_df)

            # 5. Save restricted schema
            restricted_schema_path = subset_dir / "graphGroundTruth.pkl"
            with open(restricted_schema_path, "wb") as f:
                pickle.dump(restricted_G, f)

            # 6. Save schema edges for inspection
            '''edge_path = subset_dir / "restricted_schema_edges.csv"
            pd.DataFrame(
                list(restricted_G.edges()),
                columns=["parent", "child"]
            ).to_csv(edge_path, index=False)
            '''

            # 7. Save summary
            summary = {
                "ratio": ratio,
                "ratio_name": f"{ratio_name}%",
                "repeat_id": repeat_id,
                "seed": seed,
                "num_sampled_tables": len(sampled_df),
                "num_schema_types": restricted_G.number_of_nodes(),
                "num_schema_edges": restricted_G.number_of_edges(),
            }

            summary_path = subset_dir / "summary.csv"
            pd.DataFrame([summary]).to_csv(summary_path, index=False)

            all_summaries.append(summary)

            print(
                f"[Done] {ratio_name}%, seed={seed}: "
                f"{len(sampled_df)} tables, "
                f"{restricted_G.number_of_nodes()} types, "
                f"{restricted_G.number_of_edges()} edges"
            )

    # Save global summary
    pd.DataFrame(all_summaries).to_csv(
        output_dir / "all_subset_summary.csv",
        index=False
    )


dataset = "OD_Small"
absolute_path = "E:/Project/CurrentDataset/datasets"


create_schema_size_subsets(
    mapping_csv=os.path.join(absolute_path,dataset, "groundTruth.csv"),
    column_mapping_csv = os.path.join(absolute_path,dataset, "column_gt.csv"),
    schema_path=os.path.join(absolute_path,dataset, "graphGroundTruth.pkl"),
    ori_tab_path =os.path.join(absolute_path,dataset, "Test"),
    output_dir=f"datasets/AddedExp/schemaSize",
    ratios=(0.25, 0.50, 0.75, 1.0),
    num_repeats=3,
    base_seed=42,
)