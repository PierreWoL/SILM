import argparse
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import networkx as nx
import pandas as pd
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
for path in (str(PROJECT_DIR), str(CURRENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Prompt.Relationship import (
    MostSpecificTypes,
    NE_prompt,
    RelatiosnhipPredicate,
    TopKFilter,
)
from utils.folder import mkdir


NONE_VALUES = {"", "none", "null", "nan", "n/a", "na", "no", "not applicable"}
OPENAI_MODEL_ALIASES = {
    "gpt": "gpt-3.5-turbo-1106",
    "gpt3": "gpt-3.5-turbo-1106",
    "gpt3.5": "gpt-3.5-turbo-1106",
    "gpt4": "gpt-4-1106-preview",
}
TRANSFORMER_MODEL_ALIASES = {
    "llama7b": "Llama-3.1-8B",
    "qwen14b": "Qwen/Qwen2.5-14B",
    "qwen32b": "Qwen/Qwen2.5-32B",
}


def clean_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"^```(?:\w+)?|```$", "", text).strip()
    return text


def is_none_answer(text):
    return clean_text(text).lower().strip(" .") in NONE_VALUES


def parse_type_list(answer, candidates, topk):
    if is_none_answer(answer):
        return []
    parts = re.split(r"[,;\n]+", clean_text(answer))
    selected = []
    for part in parts:
        matched = match_candidate(part, candidates)
        if matched is not None and matched not in selected:
            selected.append(matched)
        if len(selected) >= topk:
            break
    return selected


def match_candidate(answer, candidates):
    answer = clean_text(answer).strip(" .\"'")
    answer = re.sub(r"^\d+\s*[\).\-\:]\s*", "", answer).strip(" .\"'")
    if not answer:
        return None
    candidate_by_lower = {str(candidate).lower(): candidate for candidate in candidates}
    if answer.lower() in candidate_by_lower:
        return candidate_by_lower[answer.lower()]
    for candidate in candidates:
        if str(candidate).lower() == answer.lower():
            return candidate
    return None


def load_graph(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)


def get_roots(graph):
    if "Thing" in graph:
        successors = list(graph.successors("Thing"))
        if successors:
            return successors
    return [node for node in graph.nodes if graph.in_degree(node) == 0]


def node_depth(graph, node):
    roots = get_roots(graph)
    depths = []
    for root in roots:
        if root == node:
            depths.append(0)
        elif nx.has_path(graph, root, node):
            depths.append(nx.shortest_path_length(graph, root, node))
    return max(depths) if depths else 0


def get_top_level_types(graph):
    return sorted(str(node) for node in get_roots(graph) if str(node) != "Thing")


def get_descendant_types(graph, top_type):
    if top_type not in graph:
        return []
    descendants = nx.descendants(graph, top_type)
    candidates = {top_type, *descendants}
    return sorted(candidates, key=lambda node: (node_depth(graph, node), str(node)))


def normalize_table_id(table_id, table_names=None):
    table_id = str(table_id).strip()
    if table_names and table_id in table_names:
        return table_id
    if table_names and f"{table_id}.csv" in table_names:
        return f"{table_id}.csv"
    return table_id


def parse_column_id(column_id, table_names):
    column_id = str(column_id).strip()
    for table_name in sorted(table_names, key=len, reverse=True):
        if column_id.startswith(f"{table_name}."):
            return table_name, column_id[len(table_name) + 1 :]
        if table_name.lower().endswith(".csv"):
            stem = table_name[:-4]
            if column_id.startswith(f"{stem}."):
                return table_name, column_id[len(stem) + 1 :]

    if ".csv." in column_id:
        table_id, col_name = column_id.split(".csv.", 1)
        return normalize_table_id(f"{table_id}.csv", table_names), col_name

    table_id, sep, col_name = column_id.partition(".")
    if sep:
        return normalize_table_id(table_id, table_names), col_name
    return normalize_table_id(column_id, table_names), ""


def load_table_names(dataset_root, dataset):
    test_dir = Path(dataset_root) / dataset / "Test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Cannot find dataset Test folder: {test_dir}")
    return {path.name for path in test_dir.glob("*.csv")}


def load_ground_truth_table_types(dataset_root, dataset):
    gt_path = Path(dataset_root) / dataset / "groundTruth.csv"
    if not gt_path.exists():
        return {}
    gt = pd.read_csv(gt_path)
    if "fileName" not in gt.columns:
        return {}
    type_column = "class" if "class" in gt.columns else "superclass"
    if type_column not in gt.columns:
        return {}
    return {
        str(row["fileName"]): str(row[type_column])
        for _, row in gt.iterrows()
        if pd.notna(row["fileName"]) and pd.notna(row[type_column])
    }


def build_table_to_source_type(type_graph, fallback_types):
    best = {}
    for node, data in type_graph.nodes(data=True):
        for table_id in data.get("tables", []):
            table_id = str(table_id)
            score = node_depth(type_graph, node)
            old = best.get(table_id)
            if old is None or score > old[0]:
                best[table_id] = (score, str(node))
    table_to_type = {table_id: source_type for table_id, (_, source_type) in best.items()}
    for table_id, source_type in fallback_types.items():
        table_to_type.setdefault(table_id, source_type)
    return table_to_type


def get_specific_attribute_columns(attribute_graph, node, use_inherited):
    columns = set(attribute_graph.nodes[node].get("columns", []))
    if use_inherited:
        return columns
    descendant_columns = set()
    for descendant in nx.descendants(attribute_graph, node):
        descendant_columns.update(attribute_graph.nodes[descendant].get("columns", []))
    return columns - descendant_columns


def build_attribute_groups(attribute_graph, table_to_source_type, table_names, use_inherited):
    grouped = defaultdict(set)
    for attr in attribute_graph.nodes:
        columns = get_specific_attribute_columns(attribute_graph, attr, use_inherited)
        if not columns:
            continue
        for column_id in columns:
            table_id, _ = parse_column_id(column_id, table_names)
            source_type = table_to_source_type.get(table_id)
            if source_type is None and table_id.lower().endswith(".csv"):
                source_type = table_to_source_type.get(table_id[:-4])
            if source_type is None:
                continue
            grouped[(source_type, str(attr))].add(str(column_id))

    return [
        {"source": source, "attribute": attr, "columns": sorted(columns)}
        for (source, attr), columns in grouped.items()
    ]


def get_column_series(df, column_name):
    if column_name in df.columns:
        return df[column_name]
    if column_name.isdigit():
        index = int(column_name)
        if 0 <= index < len(df.columns):
            return df[df.columns[index]]
    return None


def sample_attribute_values(dataset_root, dataset, column_ids, table_names, sample_size, rng):
    values = []
    test_dir = Path(dataset_root) / dataset / "Test"
    table_cache = {}
    for column_id in column_ids:
        table_id, column_name = parse_column_id(column_id, table_names)
        table_path = test_dir / table_id
        if table_path not in table_cache:
            try:
                table_cache[table_path] = pd.read_csv(table_path, encoding="latin1")
            except Exception as exc:
                print(f"Skip table {table_path}: {exc}")
                table_cache[table_path] = None
        df = table_cache[table_path]
        if df is None:
            continue
        series = get_column_series(df, column_name)
        if series is None:
            continue
        for value in series.dropna().astype(str):
            value = value.strip()
            if value.lower() not in NONE_VALUES:
                values.append(value)

    unique_values = list(dict.fromkeys(values))
    if len(unique_values) <= sample_size:
        return unique_values
    return rng.sample(unique_values, sample_size)


def build_llm(
    model_name,
    api_key=None,
    hf_token=None,
    max_new_tokens=256,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",
):
    if "gpt" in model_name.lower():
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass --api_key when using a GPT model.")
        return {
            "provider": "openai",
            "client": OpenAI(api_key=key),
            "model": OPENAI_MODEL_ALIASES.get(model_name.lower(), model_name),
            "max_new_tokens": max_new_tokens,
        }



    model_id = TRANSFORMER_MODEL_ALIASES.get(model_name.lower(), model_name)
    token = hf_token or os.environ.get("HF_TOKEN")
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=None,
        )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if token:
        model_kwargs["token"] = token

    tokenizer_kwargs = {"padding_side": "left"}
    if token:
        tokenizer_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return {
        "provider": "transformers",
        "tokenizer": tokenizer,
        "model": model,
        "max_new_tokens": max_new_tokens,
    }


def ask_llm(llm, prompt):
    messages = [{"role": "user", "content": prompt}]
    if isinstance(llm, dict) and llm.get("provider") == "openai":
        response = llm["client"].chat.completions.create(
            model=llm["model"],
            messages=messages,
            temperature=0.1,
            max_tokens=llm["max_new_tokens"],
        )
        return clean_text(response.choices[0].message.content)
    if isinstance(llm, dict) and llm.get("provider") == "transformers":
        import torch

        tokenizer = llm["tokenizer"]
        model = llm["model"]
        prompt_token = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([prompt_token], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                cache_implementation="static",
                max_new_tokens=llm["max_new_tokens"],
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_text(decoded)
    raise TypeError(f"Unsupported LLM object: {type(llm)}")


def is_named_entity_attribute(llm, values, skip_filter):
    if skip_filter:
        return True
    prompt = NE_prompt.render(values=", ".join(values))
    answer = ask_llm(llm, prompt)
    return answer.lower().startswith("yes")


def infer_relationship_for_attribute(llm, source_type, attr_name, values, type_graph, topk, skip_ne_filter):
    if not values:
        return None
    if not is_named_entity_attribute(llm, values, skip_ne_filter):
        return None

    top_types = get_top_level_types(type_graph)
    top_prompt = TopKFilter.render(
        name=attr_name,
        attrs=values,
        types=top_types,
        topk=topk,
    )
    top_answer = ask_llm(llm, top_prompt)
    selected_top_types = parse_type_list(top_answer, top_types, topk)
    if not selected_top_types:
        return None

    descendant_types = []
    for top_type in selected_top_types:
        descendant_types.extend(get_descendant_types(type_graph, top_type))
    descendant_types = list(dict.fromkeys(descendant_types))
    if not descendant_types:
        return None

    specific_prompt = MostSpecificTypes.render(
        name=attr_name,
        attrs=values,
        types=descendant_types,
    )
    specific_answer = ask_llm(llm, specific_prompt)
    target_type = match_candidate(specific_answer, descendant_types)
    if target_type is None:
        return None

    predicate_prompt = RelatiosnhipPredicate.render(
        source=source_type,
        target=target_type,
        attr=attr_name,
    )
    predicate = ask_llm(llm, predicate_prompt)
    if not predicate or is_none_answer(predicate):
        return None

    return {
        "source": source_type,
        "attribute": attr_name,
        "target": target_type,
        "predicate": predicate,
        "top_level_candidates": selected_top_types,
        "sample_values": values,
    }


def load_computed(output_file):
    computed = set()
    if not Path(output_file).exists():
        return computed
    with open(output_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            computed.add((item["source"], item["attribute"]))
    return computed


def model_name_for_path(model_name):
    model_name = str(model_name).strip()
    return re.sub(r'[\\/:*?"<>|]+', "_", model_name)


def infer_relationships(args):
    rng = random.Random(args.seed)
    type_graph = load_graph(args.type_graph)
    attribute_graph = load_graph(args.attribute_graph)
    table_names = load_table_names(args.dataset_root, args.dataset)
    fallback_types = load_ground_truth_table_types(args.dataset_root, args.dataset)
    table_to_source_type = build_table_to_source_type(type_graph, fallback_types)
    attribute_groups = build_attribute_groups(
        attribute_graph,
        table_to_source_type,
        table_names,
        args.use_inherited_attribute_columns,
    )
    if args.limit > 0:
        attribute_groups = attribute_groups[: args.limit]

    mkdir(args.output_dir)
    output_file = Path(args.output_dir) / "relationships.jsonl"
    computed = load_computed(output_file)
    llm = build_llm(
        args.model,
        api_key=args.api_key,
        hf_token=args.hf_token,
        max_new_tokens=args.max_new_tokens,
        load_in_8bit=not args.no_8bit,
        attn_implementation=args.attn_implementation,
    )

    print(f"Loaded {len(attribute_groups)} source-attribute groups")
    print(f"Loaded {len(computed)} computed relationships")
    with open(output_file, "a", encoding="utf-8") as fout:
        for group in tqdm(attribute_groups, desc="Relationship inference"):
            key = (group["source"], group["attribute"])
            if key in computed:
                continue
            values = sample_attribute_values(
                args.dataset_root,
                args.dataset,
                group["columns"],
                table_names,
                args.sample_size,
                rng,
            )
            result = infer_relationship_for_attribute(
                llm,
                group["source"],
                group["attribute"],
                values,
                type_graph,
                args.topk,
                args.skip_ne_filter,
            )
            if result is None:
                continue
            if not args.allow_self_relationship and result["source"] == result["target"]:
                continue
            relationship_record = {
                "source": result["source"],
                "attribute": result["attribute"],
                "target": result["target"],
                "predicate": result["predicate"],
                "top_level_candidates": result["top_level_candidates"],
                "sample_values": result["sample_values"],
                "columns": group["columns"],
            }
            fout.write(json.dumps(relationship_record, ensure_ascii=False) + "\n")
            fout.flush()
            computed.add(key)

    print(f"Saved relationships to {output_file}")


def default_result_path(result_root, dataset, layer, shot, result_model, filename):
    return str(Path(result_root) / dataset / layer / str(shot) / result_model / filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer GeSI Step 3 relationships between inferred types.")
    parser.add_argument("--dataset", type=str, default="WDC")
    parser.add_argument("--dataset_root", type=str, default="E:/Project/CurrentDataset/datasets")
    parser.add_argument("--result_root", type=str, default="E:/Project/CurrentDataset/result/GeSI")
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--result_model", type=str, default="gpt3")
    parser.add_argument("--model", type=str, default="qwen14b")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_8bit", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--type_graph", type=str, default=None)
    parser.add_argument("--attribute_graph", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_ne_filter", action="store_true")
    parser.add_argument("--use_inherited_attribute_columns", action="store_true")
    parser.add_argument("--allow_self_relationship", action="store_true")
    args = parser.parse_args()

    if args.type_graph is None:
        args.type_graph = default_result_path(
            args.result_root, args.dataset, "Type", args.shot, args.result_model, "treeRefine.pkl"
        )
    if args.attribute_graph is None:
        args.attribute_graph = default_result_path(
            args.result_root, args.dataset, "Attribute", args.shot, args.result_model, "treeRefine.pkl"
        )
    if args.output_dir is None:
        args.output_dir = str(
            Path(args.result_root)
            / args.dataset
            / "Relationship"
            / str(args.shot)
            / model_name_for_path(args.model)
        )
    return args


def main():
    args = parse_args()
    infer_relationships(args)


if __name__ == "__main__":
    main()
