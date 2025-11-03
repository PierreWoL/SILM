import json
import os
import re

import numpy as np
import pandas as pd
from Sampling.Features import derive_meta_features
from Sampling.columnSampling import value_frequency, random_keys_by_frequency
from Sampling.TFIDF_sampling import compute_avg_tfidf
from Sampling.Token import tokenize
from Sampling.dataTypes import determine_data_type
import re
import nltk
from nltk.tokenize import word_tokenize


def get_df_sample(df, len_context, full=False, other_col=False, max_len=8000):
    column_samples = {}
    ignore_list = ["None", 'none', 'NaN', 'nan', 'N/A', 'na', '']
    for col in df.columns:
        sample_list = list(
            set(p[:max_len // (len_context * 3)] for p in pd.unique(df.astype(str)[col]) if p not in ignore_list))
        if full:
            meta_features = derive_meta_features(df[col])
            meta_features['rolling-mean-window-4'] = meta_features['rolling-mean-window-4'][:5]
        # Sampling from other columns
        if other_col:
            sample_list_fill_size = len_context - len(sample_list)
            nc = len(df.columns)
            per_column_context = max(1, sample_list_fill_size // nc)
            for idx, oc in enumerate(df.columns):
                items = df[oc].astype(str).iloc[0:per_column_context].tolist()
                sample_list = sample_list + ["OC: " + str(item) for item in items]
        if not sample_list:
            sample_list = ["None"]
        if len(sample_list) < len_context:
            sample_list = sample_list * len_context
        if len(sample_list) > len_context:
            sample_list = sample_list[:len_context]
        assert len(sample_list) == len_context, "An index in val_indices is length " + str(len(sample_list))
        if full:
            if meta_features['std'] == "N/A":
                sample_list = sample_list + ["" for k, v in meta_features.items()]
            else:
                sample_list = sample_list + [str(k) + ": " + str(v) for k, v in meta_features.items()]
        column_samples[col] = sample_list

    return pd.DataFrame.from_dict(column_samples)


def randomlySelect(df_column, sample_size=5):
    """
    randomly select [sample_size] unique tokens from the column
    """
    tokenized_values = df_column.apply(tokenize).explode()
    unique_tokens = tokenized_values.unique()
    return np.random.choice(unique_tokens, size=sample_size, replace=False)


def weighted_random_selection(column: pd.Series, num_samples):
    avg_tfidf_scores = compute_avg_tfidf(column)
    entries = list(avg_tfidf_scores.keys())
    weights = list(avg_tfidf_scores.values())

    try:
        selected_entries = np.random.choice(entries, size=num_samples, replace=False,
                                            p=np.array(weights) / np.sum(weights))
        return selected_entries.tolist()
    except ValueError as e:
        print("Error in selection process:", e)
        return []


def sample_column(col: pd.Series, sample=5, head=False):
    col_dict = value_frequency(col)
    if sample >= len(col):
        return col_dict, list(col)
    elif len(col_dict) <= sample:
        return col_dict, col_dict.keys()
    else:
        if head is True:
            return col_dict, list(col.head(5))
        else:
            sample_dict = random_keys_by_frequency(col, num_samples=sample)
            return col_dict, list(sample_dict.keys())


def column_context(column: pd.Series, fraction=5):
    if fraction >= len(column):
        return column
    else:
        col_dict, sample_cells = sample_column(column, sample=fraction, head=True)
    Meta_info = ""
    col_name = column.name
    sample_cells_text = f"{col_name}: {', '.join(map(str, sample_cells))}"
    if pd.api.types.is_numeric_dtype(column) is True:
        Meta_info = derive_meta_features(column)
    if Meta_info != "":
        context_column = f"{col_name}: {sample_cells_text}, Meta_info: {Meta_info}"
    else:
        context_column = f"{col_name}: {sample_cells_text}"
    return context_column


def colMetadata(col: pd.Series, sample=5):
    # Step 1: Check the Column Header or Name
    header = col.name
    # Step 2: Examine the Data Type (Text, Number, Boolean, Date, DateTime, Time)
    dataType = determine_data_type(col)
    # Step 3: Evaluate Value Uniqueness and Cardinality and
    total_length = len(col)
    # Step 4: Assess Length and Structure of Values
    features = derive_meta_features(col)
    characteristics = "cell length" if dataType != "Number" else "cell"
    if dataType == "Number":
        col_dict, sample_cells = sample_column(col, sample=sample, head=True)
    else:
        col_dict, sample_cells = sample_column(col, sample=sample)
    cardinality = len(col_dict)
    sample_cell_distribution = ""
    if dataType == "Text":
        for i, fre in col_dict.items():
            if i in sample_cells:
                sample_cell_distribution += f"{i}: {fre}, "
        # sample_cell_distribution = f"7. sample cells frequency distribution: {sample_cell_distribution}\n"

    # Step 5: Analyze Frequency Distribution and Repetition (col_dict). here we only show the sampled cells
    context = (f"1. Column Header: {header}\n"
               f"2. sampled cells: {sample_cells}\n"
               f"3. dataType: {dataType}\n"
               f"4. # unique cell values: {cardinality}\n"
               f"5.# cells: {total_length}\n"
               f"6. {characteristics} features:{features}\n"

               )
    return context


def is_website_url_column(df, column_name):
    url_pattern = re.compile(
        r'^(https?:\/\/)?(www\.)?[\da-z\.-]+\.[a-z]{2,6}(:\d+)?(\/[\w\-.~:/?#[\]@!$&\'()*+,;=]*)?$',
        re.IGNORECASE
    )

    non_null_count = df[column_name].count()
    if non_null_count == 0:
        return False
    sample_size = max(1, min(5, non_null_count))
    sample_texts = df[column_name].dropna().astype(str).sample(sample_size, replace=False)
    return all(url_pattern.match(text) for text in sample_texts)


def is_long_text_column(df, column_name, threshold=25):
    sample_texts = df[column_name].dropna().astype(str)
    avg_token_count = sample_texts.apply(lambda x: len(word_tokenize(x))).mean()
    return avg_token_count > threshold


def is_numeric_column(df, column_name):
    return df[column_name].dtype in ['int64', 'float64']


def simplify_long_text(df, column_name):
    df[column_name] = "[...]"
    return df


def truncate_long_text(df, column_name, max_words=3):
    df[column_name] = df[column_name].astype(str).apply(
        lambda x: " ".join(x.split()[:max_words]) + "..." if len(x.split()) > max_words else x
    )
    return df


def remove_empty_or_sparse_columns(df, threshold=0.9):
    non_null_ratio = df.notna().mean()
    cleaned_df = df.loc[:, non_null_ratio > (1 - threshold)]
    return cleaned_df


def simplify_columns(df):
    columns_to_sim_w = [col for col in df.columns if
                        is_website_url_column(df, col)]
    columns_to_simp_l = [col for col in df.columns if
                         is_long_text_column(df, col)]
    for col in columns_to_sim_w:
        df = simplify_long_text(df, col)
    for col in columns_to_simp_l:
        df = truncate_long_text(df, col)
    return df


def reverse_columns(df):
    """
    Reverse the column order of a given DataFrame.
    """
    return df[df.columns[::-1]]


def drop_unwanted_columns(df, long_text_threshold=50):
    df_clean = df.copy()
    cols_to_drop = []
    for col in df.columns:
        drop_col = False
        series = df[col]
        if is_long_text_column(df, col, threshold=long_text_threshold):
            cols_to_drop.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            cols_to_drop.append(col)
        elif is_website_url_column(df, col):
            cols_to_drop.append(col)
        if drop_col:
            temp_df = df_clean.drop(columns=[col])
            if temp_df.shape[1] > 0:
                df_clean = temp_df
                cols_to_drop.append(col)
            else:
                continue
    return df_clean


"""
Cut off the table
"""


def is_named_entity_column(series, token_threshold=20):
    series_str = series.dropna().astype(str)
    if len(series_str) == 0:
        return False
    avg_token_count = series_str.map(lambda x: len(x.split())).mean()
    return avg_token_count < token_threshold


def keep_leftmost_named_entity_columns(df, n=2, token_threshold=20):
    named_entity_count = 0
    keep_cols = []
    for col in df.columns:
        keep_cols.append(col)
        if is_named_entity_column(df[col], token_threshold):
            named_entity_count += 1
            if named_entity_count == n:
                break
    return df[keep_cols]


def transform_data(df, sampling_s=0):
    if sampling_s == 1:
        return reverse_columns(df)
    elif sampling_s == 2:
        return simplify_columns(df)
    elif sampling_s == 3:
        return drop_unwanted_columns(df)
    elif sampling_s == 4:
        return keep_leftmost_named_entity_columns(df, 4)
    elif sampling_s == 5:
        cut_off = keep_leftmost_named_entity_columns(df, 4)
        return drop_unwanted_columns(cut_off)
    else:
        return df


def example_data(df, method='simple', sampling_s=0, sample_size=10, summ_stats=False, other_col=False, MAX_LEN=4000):
    # df = remove_empty_or_sparse_columns(df)
    tem_df = transform_data(df, sampling_s=sampling_s)
    if len(tem_df) <= sample_size:
        return df
    else:
        if method == 'standard':
            sample_df = get_df_sample(tem_df, len_context=sample_size, full=summ_stats, other_col=other_col,
                                      max_len=MAX_LEN)
        elif method == 'simple':
            sample_df = tem_df.sample(n=sample_size)
        elif method == 'head':
            sample_df = tem_df.head(sample_size)
        else:
            sample_df = tem_df
        return sample_df


def load(dataset, sample_size=10, sample_s=0, summ_stats=False, other_col=False, MAX_LEN=4000):
    json_datasets = []
    reference_path = f"E:/Project/datasets/{dataset}/"
    gt_csv = pd.read_csv(os.path.join(reference_path, "groundTruth.csv"))
    number = 0
    for table_name in gt_csv["fileName"]:
        if table_name in os.listdir(os.path.join(reference_path, "Test")):
            matching_row = gt_csv[gt_csv["fileName"] == table_name]
            class_value = matching_row.iloc[0]["class"]
            data = pd.read_csv(os.path.join(reference_path, "Test", table_name))
            sample_df = example_data(data,
                                     method='simple', sampling_s=sample_s,
                                     sample_size=sample_size,
                                     summ_stats=summ_stats, other_col=other_col, MAX_LEN=MAX_LEN)
            json_datasets.append({'id': table_name, 'table': sample_df, 'type': class_value})
            number += 1
    return json_datasets


def loadTA(dataset, TA_PATH):
    reference_path = f"E:/Project/datasets/{dataset}/"
    with open(TA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    gt_csv = pd.read_csv(os.path.join(reference_path, "groundTruth.csv"))
    class_dict = {}
    number = 0
    for data_item in data:
        table_name = data_item["id"]
        table_attrs = data_item["attrs"]
        matching_row = gt_csv[gt_csv["fileName"] == table_name]
        class_value = matching_row.iloc[0]["class"]
        if class_value in class_dict:
            class_dict[class_value].extend(table_attrs)
        else:
            class_dict[class_value] = table_attrs
        number += 1
    for key in class_dict.keys():
        unique_list = list(set([x.lower() for x in class_dict[key]]))
        class_dict[key] = unique_list
    print(class_dict)
    return class_dict


def loadTAFilter(dataset, TA_PATH, filter=0):
    reference_path = f"E:/Project/datasets/{dataset}/"
    with open(TA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    gt_csv = pd.read_csv(os.path.join(reference_path, "groundTruth.csv"))
    class_dict = {}
    for data_item in data:
        table_name = data_item["id"]
        table_attrs = data_item["attrs"]
        matching_row = gt_csv[gt_csv["fileName"] == table_name]
        class_value = matching_row.iloc[0]["class"]
        table_df = pd.read_csv(os.path.join(f"E:/Project/datasets/{dataset}/Test/", table_name))
        cols = table_df.columns
        if class_value in class_dict:
            for index, attr in enumerate(table_attrs):
                attr_lower = attr.lower()
                if attr in class_dict[class_value]:
                    class_dict[class_value][attr_lower].append(f"{table_name[:-4]}.{cols[index]}")
                else:
                    class_dict[class_value][attr_lower] = [f"{table_name[:-4]}.{cols[index]}"]
        else:
            class_dict[class_value] = {}
            for index, attr in enumerate(table_attrs):
                attr_lower = attr.lower()
                class_dict[class_value][attr_lower] = [f"{table_name[:-4]}.{cols[index]}"]
    if filter > 0:
        for class_value in class_dict:
            attrs = class_dict[class_value].keys()
            del_attrs = []
            for attr in attrs:
                if len(class_dict[class_value][attr])<=filter:
                    del_attrs.append(attr)
            for key in del_attrs:
                class_dict[class_value].pop(key, None)  # 安全删除
    #print(class_dict)
    for class_value in class_dict:
        print(class_value, class_dict[class_value].keys())
    return class_dict
