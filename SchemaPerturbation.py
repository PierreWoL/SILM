import os
import re
import random
import math
import pandas as pd
from pathlib import Path
import hashlib
import shutil



"""
This file is copied from valentine's repo
"""
def abbreviate(name: str):
    abbreviation = []
    name = ''.join([w[0].upper()+w[1:] for w in name.split(' ')])
    words = name.split('_')
    if len(words) != 1:
        for w in words:
            if len(w)>=1:
                bound = random.randint(math.ceil(len(w)/4),math.ceil(len(w)/2))
            else:
                bound = 1
            abbreviation += w[:bound]
        abbreviation = ''.join(abbreviation)
        abbreviation = abbreviation.upper()

    else:
        capitals = re.findall('^[a-z]+|[A-Z][a-z]*|\d+', name)
        if len(capitals) >= 1:
            for c in capitals:
                if len(c) > 1:
                    bound = random.randint(math.ceil(len(c)/4),math.ceil(len(c)/2))
                else:
                    bound = 1
                abbreviation += [c[:bound]]
            abbreviation = ''.join(abbreviation)

    return abbreviation

def augment(name: str, table_name: str):
    if(len(name.split(' '))>1):
        name = ''.join([w[0].upper() + w[1:] if len(w)>1 else w[0].upper() for w in name.split(' ')])
    return table_name.split('_')[0]+'_'+name

def drop_vowels(word: str):
    word = ''.join([w[0].upper() + w[1:] if len(w) > 1 else w[0].upper() for w in word.split(' ')])
    table = str.maketrans(dict.fromkeys('aeiouyAEIOUY'))
    return word.translate(table).lower()


def perturb_dataframe_columns(
    df: pd.DataFrame,
    choice: int,
    noise_ratio: float = 1.0,
    seed: int = 42,
    inplace: bool = False,
    return_records: bool = False
):
    """
    Randomly select n% columns from a table and perturb only those selected columns.

    choice = 2: abbreviation
    choice = 3: drop vowels

    noise_ratio:
        1.0  = perturb all columns
        0.25 = randomly perturb 25% columns per table
        0.50 = randomly perturb 50% columns per table
        0.75 = randomly perturb 75% columns per table
    """

    new_df = df if inplace else df.copy()

    old_columns = list(new_df.columns)
    num_cols = len(old_columns)

    if num_cols == 0 or noise_ratio <= 0:
        if return_records:
            return new_df, []
        return new_df

    rng = random.Random(seed)

    # number of columns selected for perturbation
    k = math.ceil(num_cols * noise_ratio)
    k = max(1, min(k, num_cols))

    selected_indices = set(rng.sample(range(num_cols), k))

    new_columns = []
    records = []

    for idx, col in enumerate(old_columns):
        col_str = str(col)
        # only perturb selected columns
        if idx in selected_indices:
            if choice == 1:
                action = rng.choice([2, 3])
            else:
                action = choice

            if action == 2:
                new_col = abbreviate(col_str)
                if new_col == "" or new_col == col_str:
                    new_col = drop_vowels(col_str)

            elif action == 3:
                new_col = drop_vowels(col_str)

            else:
                new_col = col_str

            # avoid duplicate column headers after perturbation
            if new_col in new_columns or new_col == "":
                final_col = col_str
            else:
                final_col = new_col

        else:
            final_col = col_str

        new_columns.append(final_col)

        records.append({
            "column_index": idx,
            "old_column": col_str,
            "new_column": final_col,
            "selected_for_noise": idx in selected_indices,
            "actually_changed": final_col != col_str
        })

    new_df.columns = new_columns

    if return_records:
        return new_df, records

    return new_df






def stable_seed(base_seed: int, file_name: str) -> int:
    """
    Generate a stable per-table seed.
    Avoid using Python's built-in hash(), because it is not stable across runs.
    """
    text = f"{base_seed}_{file_name}"
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def perturb_tables_and_update_column_gt(
    table_dir: str,
    output_table_dir: str,
    column_gt_path: str,
    output_column_gt_path: str,
    choice: int,
    noise_ratio: float,
    seed: int = 42,
    file_list=None,
    table_sep: str = ",",
    gt_sep: str = ",",
    encoding: str = "utf-8",
    save_mapping_path: str = None,
    save_update_report_path: str = None,
):
    """
    Perturb n% column headers for each input table and update column_gt.csv accordingly.

    Parameters
    ----------
    table_dir:
        Folder containing original input tables.

    output_table_dir:
        Folder to save perturbed tables.

    column_gt_path:
        Path to the original column_gt.csv.

    output_column_gt_path:
        Path to save the updated column_gt.csv.

    choice:
        2 = abbreviation
        3 = drop vowels

    noise_ratio:
        Ratio of columns to perturb per table.
        Example:
            0.25 = perturb 25% columns per table
            0.50 = perturb 50% columns per table
            0.75 = perturb 75% columns per table
            1.00 = perturb all columns per table

    seed:
        Base random seed.

    file_list:
        Optional list of table file names to process.
        If None, all CSV files under table_dir will be processed.

    table_sep:
        Separator for input/output tables.

    gt_sep:
        Separator for column_gt.csv.

    save_mapping_path:
        Optional path to save column-level perturbation mapping.

    save_update_report_path:
        Optional path to save column_gt update report.
    """

    table_dir = Path(table_dir)
    output_table_dir = Path(output_table_dir)
    output_table_dir.mkdir(parents=True, exist_ok=True)

    column_gt_path = Path(column_gt_path)
    output_column_gt_path = Path(output_column_gt_path)
    output_column_gt_path.parent.mkdir(parents=True, exist_ok=True)

    # Load column ground truth
    column_gt = pd.read_csv(column_gt_path, sep=gt_sep, encoding=encoding)

    required_cols = {"fileName", "colName"}
    missing_cols = required_cols - set(column_gt.columns)
    if missing_cols:
        raise ValueError(f"column_gt.csv is missing required columns: {missing_cols}")

    # Decide which tables to process
    if file_list is None:
        table_paths = sorted(table_dir.glob("*.csv"))
    else:
        table_paths = [table_dir / f for f in file_list]

    all_records = []

    for table_path in table_paths:
        if not table_path.exists():
            print(f"[Warning] Table not found, skipped: {table_path}")
            continue

        file_name = table_path.name

        # Use a different deterministic seed for each table
        table_seed = stable_seed(seed, file_name)

        # Important:
        # Your abbreviate() function uses random.randint() directly,
        # so we also seed the global random module here.
        random.seed(table_seed)

        df = pd.read_csv(table_path, sep=table_sep, encoding=encoding)

        perturbed_df, records = perturb_dataframe_columns(
            df=df,
            choice=choice,
            noise_ratio=noise_ratio,
            seed=table_seed,
            inplace=False,
            return_records=True,
        )

        # Save perturbed table with the same file name
        output_table_path = output_table_dir / file_name
        perturbed_df.to_csv(output_table_path, index=False, sep=table_sep, encoding=encoding)

        # Add file-level information to each column record
        for r in records:
            r["fileName"] = file_name
            r["choice"] = choice
            r["noise_ratio"] = noise_ratio
            r["table_seed"] = table_seed

        all_records.extend(records)

    mapping_df = pd.DataFrame(all_records)

    # Update column_gt.csv
    updated_column_gt = column_gt.copy()

    # Make matching robust
    updated_column_gt["fileName"] = updated_column_gt["fileName"].astype(str)
    updated_column_gt["colName"] = updated_column_gt["colName"].astype(str)

    update_reports = []

    if not mapping_df.empty:
        changed_mapping = mapping_df[mapping_df["actually_changed"] == True]

        for _, row in changed_mapping.iterrows():
            file_name = str(row["fileName"])
            old_col = str(row["old_column"])
            new_col = str(row["new_column"])

            mask = (
                (updated_column_gt["fileName"] == file_name)
                & (updated_column_gt["colName"] == old_col)
            )

            matched_rows = int(mask.sum())

            if matched_rows > 0:
                updated_column_gt.loc[mask, "colName"] = new_col

            update_reports.append({
                "fileName": file_name,
                "old_column": old_col,
                "new_column": new_col,
                "gt_rows_matched": matched_rows,
                "updated": matched_rows > 0,
            })

    update_report_df = pd.DataFrame(update_reports)

    # Save updated column_gt
    updated_column_gt.to_csv(
        output_column_gt_path,
        index=False,
        sep=gt_sep,
        encoding=encoding,
    )

    # Save mapping files
    if save_mapping_path is None:
        save_mapping_path = output_table_dir / "column_name_perturbation_mapping.csv"
    else:
        save_mapping_path = Path(save_mapping_path)

    save_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(save_mapping_path, index=False, encoding=encoding)

    if save_update_report_path is None:
        save_update_report_path = output_table_dir / "column_gt_update_report.csv"
    else:
        save_update_report_path = Path(save_update_report_path)

    save_update_report_path.parent.mkdir(parents=True, exist_ok=True)
    update_report_df.to_csv(save_update_report_path, index=False, encoding=encoding)

    print("Done.")
    print(f"Perturbed tables saved to: {output_table_dir}")
    print(f"Updated column_gt saved to: {output_column_gt_path}")
    print(f"Perturbation mapping saved to: {save_mapping_path}")
    print(f"Update report saved to: {save_update_report_path}")

    if not update_report_df.empty:
        unmatched = update_report_df[update_report_df["gt_rows_matched"] == 0]
        if len(unmatched) > 0:
            print(f"[Warning] {len(unmatched)} changed columns were not found in column_gt.csv.")
            print("Please check column_gt_update_report.csv.")

    return updated_column_gt, mapping_df, update_report_df



dataset = "WDC"
abs_path = "E:/Project/CurrentDataset/datasets/"
output_path = f"datasets/AddedExp/noiseLevel/"
noise_levels = [0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.8]
for noise_ration in noise_levels:
    updated_gt, mapping_df, update_report_df = perturb_tables_and_update_column_gt(
        table_dir=os.path.join(abs_path, dataset, "Test"),
        output_table_dir=os.path.join(output_path, f"{str(int(noise_ration * 100)) + "_pct"}", "Test"),
        column_gt_path=os.path.join(abs_path, dataset, "column_gt.csv"),
        output_column_gt_path=os.path.join(output_path, f"{str(int(noise_ration * 100)) + "_pct"}", "column_gt.csv"),
        choice=2,  # 1 = random choice, 2 = abbreviation, 3 = drop vowels
        noise_ratio=noise_ration,  # perturb 25% columns per table
        seed=42,
        encoding="latin1"
    )
    src_file_gt_table = os.path.join(abs_path,dataset,  "groundTruth.csv")
    src_file_graph = os.path.join(abs_path, dataset,  "graphGroundTruth.pkl")

    check_graph = Path(os.path.join(output_path, f"{str(int(noise_ration * 100)) + "_pct"}", "graphGroundTruth.pkl"))
    check_gt = Path(os.path.join(output_path, f"{str(int(noise_ration * 100)) + "_pct"}", "groundTruth.csv"))

    if not check_gt.exists():
        shutil.copy(src_file_gt_table, os.path.join(output_path, f"{str(int(noise_ration * 100)) + "_pct"}"))

    if not check_graph.exists():
        shutil.copy(src_file_graph,  os.path.join(output_path, f"{str(int(noise_ration * 100)) + "_pct"}"))

