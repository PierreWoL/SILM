import json
import os
import CA_evaluation as eva
import pandas as pd


def findTables(dataset):
    reference_path = f"datasets/{dataset}/"
    gt_csv = pd.read_csv(os.path.join(reference_path, "groundTruth.csv"))
    class_to_files = gt_csv.groupby('class')['fileName'].apply(set).to_dict()
    return class_to_files


def tableAttrMapping(table_names, dataset, TA_result):
    mapping = {}
    for tableN in table_names:
        table = pd.read_csv(f"datasets/{dataset}/Test/{tableN}")
        table_cols = [f"{tableN[:-4]}.{header}" for header in table.columns]
        table_col_attrs = TA_result[tableN]
        # print(len(table_col_attrs),len(table_cols) )
        for index, table_col_attr in enumerate(table_col_attrs):
            if table_col_attr.lower() not in mapping:
                mapping[table_col_attr.lower()] = [table_cols[index]]
            else:
                mapping[table_col_attr.lower()].append(table_cols[index])
    return mapping


def mapAttrClusters(fet_dict, TA_mapping):
    clusters = {}
    for conceptualAttri in fet_dict:
        inferred_attrs = fet_dict[conceptualAttri]
        mappingCols = []
        for attr in inferred_attrs:
            if attr in TA_mapping:
                mappingCols.extend(TA_mapping[attr])
        clusters[conceptualAttri] = mappingCols
    return clusters


def findAttrs(dataset, AR_path, TA_path):
    class_file_Dict = findTables(dataset)
    fet_dict = {}
    with open(AR_path, "r", encoding="utf-8") as f:
        AR_result = [json.loads(line) for line in f if line.strip()]
    with open(TA_path, "r", encoding="utf-8") as f:
        TA_result = {json.loads(line)["id"]: json.loads(line)["attrs"] for line in f if line.strip()}
    for fet_result in AR_result:
        tables = class_file_Dict[fet_result["class"]]
        colAttrMapping = tableAttrMapping(tables, dataset, TA_result)
        cluster_result = mapAttrClusters(fet_result["inferred"],colAttrMapping)
        print(fet_result["class"], class_file_Dict[fet_result["class"]],cluster_result)
        fet_dict[fet_result["class"]] = cluster_result
    return fet_dict
"""
ar_p = ""
ta_p = ""

dataset_name = "GDS"
model = "qwen"
filter_n = 3
result_dict = findAttrs(dataset_name,
                        f"Result/{dataset_name}/Step2/AR/1/{model}/filter{filter_n}/ARresults.jsonl",
                        f"Result/{dataset_name}/Step2/TA/split/1/{model}/TAresults.jsonl" )
target_f = f"Result/{dataset_name}/Step2/AR/1/{model}/examples/filter{filter_n}/"
os.makedirs(target_f, exist_ok=True)
eva.test_cols(dataset_name, result_dict, superClass=False,
              folder=target_f)
"""