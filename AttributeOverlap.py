import os
import ast
import shutil

import pandas as pd
import networkx as nx
from collections import defaultdict
from pathlib import Path



def parse_list_cell(x):
    """
    Parse cells such as "['Event']" into Python lists.
    If parsing fails, return an empty list.
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]
    except Exception:
        return [x]


def read_table_type_mapping(table_type_path):
    """
    Input columns:
        fileName,class,superclass

    Returns:
        table_to_type: fileName -> most-specific type
        table_to_superclasses: fileName -> list of superclass types
        table_type_df
    """
    df = pd.read_csv(table_type_path)

    required_cols = {"fileName", "class", "superclass"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in table type file: {missing}")

    df["superclass_parsed"] = df["superclass"].apply(parse_list_cell)

    table_to_type = dict(zip(df["fileName"], df["class"]))
    table_to_superclasses = dict(zip(df["fileName"], df["superclass_parsed"]))

    return table_to_type, table_to_superclasses, df


def read_table_attributes(attribute_path):
    """
    Input columns:
        fileName,colName,vals,ColumnLabel,LowestClass,TopClass

    Returns:
        attr_df
    """
    df = pd.read_csv(attribute_path)

    required_cols = {
        "fileName", "colName", "ColumnLabel", "LowestClass", "TopClass"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in attribute file: {missing}")

    df["TopClass_parsed"] = df["TopClass"].apply(parse_list_cell)

    # remove empty / invalid attribute labels
    df["ColumnLabel"] = df["ColumnLabel"].astype(str).str.strip()
    df = df[
        (df["ColumnLabel"] != "") &
        (df["ColumnLabel"].str.lower() != "nan")
    ].copy()

    return df


def build_table_to_attrs(attr_df):
    """
    Build:
        table -> set of unique ColumnLabel
    """
    table_to_attrs = (
        attr_df.groupby("fileName")["ColumnLabel"]
        .apply(lambda x: set(x.dropna().astype(str).str.strip()))
        .to_dict()
    )

    return table_to_attrs


def build_type_to_attrs_by_table_class(table_to_type, table_to_attrs):
    """
    Build type-level attributes using table-to-most-specific-type mapping.

    For each type c:
        A(c) = union of attributes from all tables whose class is c
    """
    type_to_attrs = defaultdict(set)
    type_to_tables = defaultdict(list)

    for table, type_name in table_to_type.items():
        attrs = table_to_attrs.get(table, set())

        type_to_attrs[type_name].update(attrs)
        type_to_tables[type_name].append(table)

    return dict(type_to_attrs), dict(type_to_tables)


def compute_table_attribute_overlap(table_to_type, table_to_attrs, type_to_attrs):
    """
    Compute:
        overlap(t) = |A(t) ∩ A(class(t))| / |A(class(t))|

    Since A(class(t)) is built as the union of attributes from tables of that
    class, A(t) should normally be a subset of A(class(t)).
    """
    records = []

    for table, type_name in table_to_type.items():
        table_attrs = set(table_to_attrs.get(table, set()))
        type_attrs = set(type_to_attrs.get(type_name, set()))

        if len(type_attrs) == 0:
            covered_attrs = set()
            overlap = 0.0
        else:
            covered_attrs = table_attrs.intersection(type_attrs)
            overlap = len(covered_attrs) / len(type_attrs)

        records.append({
            "fileName": table,
            "class": type_name,
            "num_table_attrs": len(table_attrs),
            "num_type_attrs": len(type_attrs),
            "num_covered_attrs": len(covered_attrs),
            "attribute_overlap": overlap,
            "table_attrs": sorted(table_attrs),
            "type_attrs": sorted(type_attrs),
        })

    return pd.DataFrame(records)


def induce_schema_from_retained_tables(
    G,
    retained_tables,
    table_to_type,
    table_to_superclasses=None,
    keep_ancestors=True
):
    """
    Build the induced schema for a retained table subset.

    Retain:
        1. most-specific types of retained tables
        2. their ancestor types in G, if keep_ancestors=True

    If G does not contain all ancestor edges, table_to_superclasses can also
    be used to retain superclass nodes.
    """
    retained_tables = set(retained_tables)

    retained_specific_types = {
        table_to_type[t]
        for t in retained_tables
        if t in table_to_type
    }

    retained_nodes = set(retained_specific_types)

    if keep_ancestors:
        for type_name in retained_specific_types:
            if type_name in G:
                retained_nodes.update(nx.ancestors(G, type_name))

        if table_to_superclasses is not None:
            for table in retained_tables:
                for superclass in table_to_superclasses.get(table, []):
                    if superclass in G:
                        retained_nodes.add(superclass)

    G_sub = G.subgraph(retained_nodes).copy()

    # Store retained tables on their most-specific type nodes
    for node in G_sub.nodes:
        G_sub.nodes[node]["tables"] = [
            table
            for table in retained_tables
            if table_to_type.get(table) == node
        ]

    return G_sub


def create_attribute_overlap_variants(
    table_type_path,
    attribute_path,
    table_ori_path,
    schema_graph,
    thresholds=(0.25, 0.50, 0.75),
    output_dir="attribute_overlap_variants"
):
    """
    Create:
        D_25 = tables with attribute_overlap >= 25%
        D_50 = tables with attribute_overlap >= 50%
        D_75 = tables with attribute_overlap >= 75%
    """

    table_to_type, table_to_superclasses, table_type_df = read_table_type_mapping(
        table_type_path
    )

    attr_df = read_table_attributes(attribute_path)

    table_to_attrs = build_table_to_attrs(attr_df)

    type_to_attrs, type_to_tables = build_type_to_attrs_by_table_class(
        table_to_type=table_to_type,
        table_to_attrs=table_to_attrs
    )

    overlap_df = compute_table_attribute_overlap(
        table_to_type=table_to_type,
        table_to_attrs=table_to_attrs,
        type_to_attrs=type_to_attrs
    )



    variants = {}

    for th in thresholds:
        variant_name = f"D_{int(th * 100)}"

        retained_df = overlap_df[
            overlap_df["attribute_overlap"] >= th
        ].copy()

        retained_tables = set(retained_df["fileName"])

        G_sub = induce_schema_from_retained_tables(
            G=schema_graph,
            retained_tables=retained_tables,
            table_to_type=table_to_type,
            table_to_superclasses=table_to_superclasses,
            keep_ancestors=True
        )

        stats = {
            "threshold": th,
            "num_tables": len(retained_tables),
            "num_types": G_sub.number_of_nodes(),
            "num_edges": G_sub.number_of_edges(),
            "avg_attribute_overlap": retained_df["attribute_overlap"].mean()
            if len(retained_df) > 0 else 0.0,
            "min_attribute_overlap": retained_df["attribute_overlap"].min()
            if len(retained_df) > 0 else 0.0,
            "max_attribute_overlap": retained_df["attribute_overlap"].max()
            if len(retained_df) > 0 else 0.0,
            "avg_num_table_attrs": retained_df["num_table_attrs"].mean()
            if len(retained_df) > 0 else 0.0,
            "avg_num_type_attrs": retained_df["num_type_attrs"].mean()
            if len(retained_df) > 0 else 0.0,
        }

        variants[variant_name] = {
            "threshold": th,
            "retained_tables": retained_tables,
            "retained_overlap_df": retained_df,
            "schema": G_sub,
            "stats": stats,
        }
        # Save retained ground truth list
        groundTruth_df = pd.read_csv(table_type_path)
        column_gt_df = pd.read_csv(attribute_path)
        gt_filtered = groundTruth_df[groundTruth_df["fileName"].isin(retained_tables)]
        column_gt_filtered  = column_gt_df[column_gt_df["fileName"].isin(retained_tables)]
        os.makedirs(os.path.join(output_dir, f"{variant_name}"), exist_ok=True)
        gt_filtered.to_csv(
            os.path.join(output_dir, f"{variant_name}","groundTruth.csv"),
            index=False)
        column_gt_filtered.to_csv(
            os.path.join(output_dir, f"{variant_name}", "column_gt.csv"),
            index=False)
        # Save table-level overlap details
        retained_df.to_csv(
            os.path.join(output_dir,f"{variant_name}","overlap_details.csv"),
            index=False
        )
        # Save files to target path
        subset_dir = os.path.join(output_dir, f"{variant_name}")
        store_dir = Path(os.path.join(subset_dir,"Test/"))
        store_dir.mkdir(parents=True, exist_ok=True)
        ## 1. Save sampled tables
        source_folder = Path(table_ori_path)
        target_folder = Path(store_dir)
        for filename in retained_tables:
            src = source_folder / filename
            dst = target_folder / filename
            if src.exists() and src.is_file():
                shutil.copy2(src, dst)  # copy2 会保留修改时间等元信息
                print(f"Copied: {filename}")
            else:
                print(f"Not found: {filename}")
        # Save induced schema
        with open(os.path.join(output_dir, f"{variant_name}","graphGroundTruth.pkl"), "wb") as f:
            pickle.dump(G_sub, f)

    stats_df = pd.DataFrame({
        name: info["stats"]
        for name, info in variants.items()
    }).T

    overlap_df.to_csv(
        os.path.join(output_dir, "all_table_attribute_overlap.csv"),
        index=False
    )

    stats_df.to_csv(
        os.path.join(output_dir, "variant_statistics.csv")
    )

    # Save type-level attributes for checking
    type_attr_records = []
    for type_name, attrs in type_to_attrs.items():
        type_attr_records.append({
            "class": type_name,
            "num_attrs": len(attrs),
            "attributes": sorted(attrs),
            "num_tables": len(type_to_tables.get(type_name, [])),
        })

    pd.DataFrame(type_attr_records).to_csv(
        os.path.join(output_dir, "type_level_attributes.csv"),
        index=False
    )

    return variants, overlap_df, stats_df, type_to_attrs


import pickle

dataset = "OD_Small"
absolute_path = "E:/Project/CurrentDataset/datasets"
with open(os.path.join(absolute_path,dataset, "graphGroundTruth.pkl"), "rb") as f:
    G = pickle.load(f)


variants, overlap_df, stats_df, type_to_attrs = create_attribute_overlap_variants(
    table_type_path=os.path.join(absolute_path,dataset,"groundTruth.csv"),
    attribute_path=os.path.join(absolute_path,dataset,"column_gt.csv"),
    table_ori_path = os.path.join(absolute_path,dataset,"Test"),
    schema_graph=G,
    thresholds=(0.25, 0.50, 0.75),
    output_dir=f"datasets/AddedExp/attribute_overlap"
)

print(stats_df)