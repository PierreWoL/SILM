import argparse
import ast
import csv
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

try:
    from langchain_ollama import OllamaLLM
except ModuleNotFoundError:
    OllamaLLM = None


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


NONE_VALUES = {"", "none", "null", "nan", "n/a", "na", "no", "not applicable"}
OLLAMA_MODEL_ALIASES = {
    "llama7b": "llama3.1:8b",
    "llama8b": "llama3.1:8b",
    "qwen14b": "qwen2.5:14b",
    "qwen32b": "qwen2.5:32b",
}
OPENAI_MODEL_ALIASES = {
    "gpt": "gpt-3.5-turbo-1106",
    "gpt3": "gpt-3.5-turbo-1106",
    "gpt3.5": "gpt-3.5-turbo-1106",
    "gpt4": "gpt-4-1106-preview",
}


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>\s*", text, flags=re.IGNORECASE)[-1].strip()
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


def require_networkx():
    if nx is None:
        raise ModuleNotFoundError(
            "networkx is required when --attribute_source attribute_graph is used. "
            "Install it with: pip install networkx"
        )


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    text = str(value).strip()
    return not text or text.lower() in NONE_VALUES


def read_csv_rows(path, encoding="utf-8-sig"):
    last_error = None
    for selected_encoding in (encoding, "latin1"):
        try:
            with open(path, "r", encoding=selected_encoding, newline="") as fin:
                reader = csv.DictReader(fin)
                rows = list(reader)
                fieldnames = [name for name in (reader.fieldnames or []) if name is not None]
                return rows, fieldnames
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def read_csv_table(path, encoding="latin1"):
    last_error = None
    for selected_encoding in (encoding, "utf-8-sig"):
        try:
            with open(path, "r", encoding=selected_encoding, newline="") as fin:
                reader = csv.reader(fin)
                headers = next(reader, [])
                rows = list(reader)
                return headers, rows
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


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
    table_names = load_table_names(dataset_root, dataset)
    table_to_types, _ = load_ground_truth_table_type_lists(dataset_root, dataset, table_names)
    return {table_id: types[0] for table_id, types in table_to_types.items() if types}


def parse_type_values(value):
    if is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = text
        if isinstance(parsed, (list, tuple, set)):
            raw_values = list(parsed)
        elif isinstance(parsed, str) and "," in parsed:
            raw_values = parsed.split(",")
        else:
            raw_values = [parsed]

    types = []
    for item in raw_values:
        type_name = str(item).strip().strip("\"'")
        if type_name and type_name.lower() not in NONE_VALUES and type_name not in types:
            types.append(type_name)
    return types


def choose_type_column(columns, preferred=None):
    if preferred and preferred in columns:
        return preferred
    for column in ("superclass", "TopClass", "topClass", "class", "LowestClass"):
        if column in columns:
            return column
    return None


def load_ground_truth_table_type_lists(dataset_root, dataset, table_names, type_column=None):
    gt_path = Path(dataset_root) / dataset / "groundTruth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Cannot find ground truth file: {gt_path}")
    rows, columns = read_csv_rows(gt_path)
    if "fileName" not in columns:
        raise ValueError(f"{gt_path} must contain a fileName column")
    selected_type_column = choose_type_column(columns, type_column)
    if selected_type_column is None:
        raise ValueError(f"{gt_path} must contain superclass, class, or another type column")

    table_to_types = defaultdict(list)
    top_level_types = []
    for row in rows:
        file_name = row.get("fileName")
        if is_missing(file_name):
            continue
        table_id = normalize_table_id(file_name, table_names)
        source_types = parse_type_values(row.get(selected_type_column))
        if not source_types and selected_type_column != "class" and "class" in columns:
            source_types = parse_type_values(row.get("class"))
        for source_type in source_types:
            if source_type not in table_to_types[table_id]:
                table_to_types[table_id].append(source_type)
            if source_type not in top_level_types:
                top_level_types.append(source_type)
    return dict(table_to_types), sorted(top_level_types)


def build_ground_truth_attribute_groups(
    dataset_root,
    dataset,
    table_names,
    type_column=None,
    column_gt_path=None,
):
    table_to_types, top_level_types = load_ground_truth_table_type_lists(
        dataset_root,
        dataset,
        table_names,
        type_column,
    )
    column_gt = Path(column_gt_path) if column_gt_path else Path(dataset_root) / dataset / "column_gt.csv"
    if not column_gt.exists():
        raise FileNotFoundError(f"Cannot find column ground truth file: {column_gt}")
    rows, columns = read_csv_rows(column_gt)
    required_columns = {"fileName", "colName"}
    missing_columns = required_columns - set(columns)
    if missing_columns:
        raise ValueError(f"{column_gt} is missing required columns: {sorted(missing_columns)}")

    attr_column = "ColumnLabel" if "ColumnLabel" in columns else "colName"
    grouped = defaultdict(set)
    source_tables = defaultdict(set)
    for row in rows:
        file_name = row.get("fileName")
        column_name = row.get("colName")
        if is_missing(file_name) or is_missing(column_name):
            continue
        table_id = normalize_table_id(file_name, table_names)
        source_types = table_to_types.get(table_id, [])
        if not source_types:
            continue
        column_name = str(column_name).strip()
        attr_name = row.get(attr_column)
        if is_missing(attr_name) or not str(attr_name).strip() or str(attr_name).strip().lower() in NONE_VALUES:
            attr_name = column_name
        attr_name = str(attr_name).strip()
        column_id = f"{table_id}.{column_name}"
        for source_type in source_types:
            grouped[(source_type, attr_name)].add(column_id)
            source_tables[(source_type, attr_name)].add(table_id)

    attribute_groups = [
        {
            "source": source,
            "attribute": attr,
            "columns": sorted(columns),
            "source_tables": sorted(source_tables[(source, attr)]),
        }
        for (source, attr), columns in sorted(grouped.items())
    ]
    return attribute_groups, top_level_types


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


def get_column_values(table, column_name):
    headers, rows = table
    if column_name in headers:
        index = headers.index(column_name)
    elif column_name.isdigit():
        index = int(column_name)
    else:
        return []
    if not 0 <= index < len(headers):
        return []
    values = []
    for row in rows:
        if index < len(row):
            values.append(row[index])
    return values


def get_column_series(df, column_name):
    if hasattr(df, "columns") and column_name in df.columns:
        return df[column_name]
    if column_name.isdigit():
        index = int(column_name)
        if hasattr(df, "columns") and 0 <= index < len(df.columns):
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
                table_cache[table_path] = read_csv_table(table_path, encoding="latin1")
            except Exception as exc:
                print(f"Skip table {table_path}: {exc}")
                table_cache[table_path] = None
        table = table_cache[table_path]
        if table is None:
            continue
        column_values = get_column_values(table, column_name)
        if not column_values:
            continue
        for value in column_values:
            value = str(value).strip()
            if not is_missing(value):
                values.append(value)

    unique_values = list(dict.fromkeys(values))
    if len(unique_values) <= sample_size:
        return unique_values
    return rng.sample(unique_values, sample_size)


def resolve_ollama_model_name(model_name):
    model_name = str(model_name).strip()
    return OLLAMA_MODEL_ALIASES.get(model_name.lower(), model_name)


def resolve_openai_model_name(model_name):
    model_name = str(model_name).strip()
    return OPENAI_MODEL_ALIASES.get(model_name.lower(), model_name)


def choose_provider(model_name, provider, use_gpt):
    if provider != "auto":
        return provider
    if use_gpt or "gpt" in str(model_name).lower():
        return "openai"
    return "ollama"


def resolve_model_name(model_name, provider):
    if provider == "openai":
        return resolve_openai_model_name(model_name)
    return resolve_ollama_model_name(model_name)


def model_name_for_path(model_name):
    model_name = str(model_name).strip()
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", model_name).strip(" ._")
    return safe_name or "ollama"


def preferred_openai_token_parameter(model_name):
    model_name = str(model_name).lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def is_unsupported_parameter_error(exc, parameter):
    text = str(exc).lower()
    parameter = parameter.lower()
    return (
        "unsupported parameter" in text
        and (f"'{parameter}'" in text or f'"{parameter}"' in text or parameter in text)
    )


def is_temperature_value_error(exc):
    text = str(exc).lower()
    return (
        "temperature" in text
        and ("unsupported" in text or "not support" in text or "only the default" in text)
    )


def openai_chat_completion(llm, messages):
    preferred_token_param = preferred_openai_token_parameter(llm["model"])
    token_params = [preferred_token_param]
    fallback_token_param = "max_completion_tokens" if preferred_token_param == "max_tokens" else "max_tokens"
    token_params.append(fallback_token_param)

    last_error = None
    for token_param in token_params:
        request = {
            "model": llm["model"],
            "messages": messages,
            "temperature": llm["temperature"],
        }
        if llm["num_predict"] > 0:
            request[token_param] = llm["num_predict"]

        while True:
            try:
                return llm["client"].chat.completions.create(**request)
            except Exception as exc:
                if "temperature" in request and (
                    is_unsupported_parameter_error(exc, "temperature")
                    or is_temperature_value_error(exc)
                ):
                    request.pop("temperature")
                    continue
                if token_param in request and is_unsupported_parameter_error(exc, token_param):
                    last_error = exc
                    break
                raise

    raise last_error


def build_llm(
    model_name,
    provider,
    use_gpt,
    ollama_base_url,
    api_key,
    openai_base_url,
    temperature,
    top_p,
    num_ctx,
    num_predict,
):
    selected_provider = choose_provider(model_name, provider, use_gpt)
    resolved_model_name = resolve_model_name(model_name, selected_provider)
    if selected_provider == "openai":
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openai is required when using the OpenAI-compatible backend. "
                "Install it with: pip install openai"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass --api_key when using a closed-source model.")
        client_kwargs = {"api_key": key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        return {
            "provider": "openai",
            "client": OpenAI(**client_kwargs),
            "model": resolved_model_name,
            "temperature": temperature,
            "num_predict": num_predict,
        }

    if OllamaLLM is None:
        raise ModuleNotFoundError(
            "langchain-ollama is required when using the Ollama backend. "
            "Install it with: pip install langchain-ollama"
        )
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_ctx": num_ctx,
    }
    if num_predict > 0:
        options["num_predict"] = num_predict
    return {
        "provider": "ollama",
        "client": OllamaLLM(
            base_url=ollama_base_url,
            model=resolved_model_name,
            options=options,
        ),
        "model": resolved_model_name,
    }


def ask_llm(llm, prompt):
    messages = [{"role": "user", "content": prompt}]
    if isinstance(llm, dict) and llm.get("provider") == "ollama":
        try:
            response = llm["client"].invoke(messages)
        except TypeError:
            response = llm["client"].invoke(prompt)
        return clean_text(response)
    if isinstance(llm, dict) and llm.get("provider") == "openai":
        response = openai_chat_completion(llm, messages)
        return clean_text(response.choices[0].message.content)
    raise TypeError(f"Unsupported LLM object: {type(llm)}")


def is_named_entity_attribute(llm, values, skip_filter):
    if skip_filter:
        return True
    prompt = NE_prompt.render(values=", ".join(values))
    answer = ask_llm(llm, prompt)
    return answer.lower().startswith("yes")


def infer_relationship_for_attribute(
    llm,
    source_type,
    attr_name,
    values,
    top_level_types,
    topk,
    skip_ne_filter,
    allow_self_relationship=False,
    type_graph=None,
    use_subtypes=False,
):
    if not values:
        return None
    if not is_named_entity_attribute(llm, values, skip_ne_filter):
        return None

    top_types = [
        candidate
        for candidate in top_level_types
        if allow_self_relationship or str(candidate) != str(source_type)
    ]
    if not top_types:
        return None
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

    if use_subtypes:
        descendant_types = []
        for top_type in selected_top_types:
            descendant_types.extend(get_descendant_types(type_graph, top_type))
        target_candidates = list(dict.fromkeys(descendant_types))
    else:
        target_candidates = selected_top_types

    specific_prompt = MostSpecificTypes.render(
        name=attr_name,
        attrs=values,
        types=target_candidates,
    )
    specific_answer = ask_llm(llm, specific_prompt)
    target_type = match_candidate(specific_answer, target_candidates)
    if target_type is None:
        parsed_targets = parse_type_list(specific_answer, target_candidates, 1)
        target_type = parsed_targets[0] if parsed_targets else None
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


def infer_relationships(args):
    rng = random.Random(args.seed)
    table_names = load_table_names(args.dataset_root, args.dataset)
    type_graph = None
    use_subtypes = False
    if args.attribute_source == "attribute_graph":
        require_networkx()
        type_graph = load_graph(args.type_graph)
        attribute_graph = load_graph(args.attribute_graph)
        fallback_types = load_ground_truth_table_types(args.dataset_root, args.dataset)
        table_to_source_type = build_table_to_source_type(type_graph, fallback_types)
        attribute_groups = build_attribute_groups(
            attribute_graph,
            table_to_source_type,
            table_names,
            args.use_inherited_attribute_columns,
        )
        top_level_types = get_top_level_types(type_graph)
        use_subtypes = True
    else:
        attribute_groups, top_level_types = build_ground_truth_attribute_groups(
            args.dataset_root,
            args.dataset,
            table_names,
            type_column=args.gt_type_column,
            column_gt_path=args.column_gt,
        )

    if args.limit > 0:
        attribute_groups = attribute_groups[: args.limit]

    mkdir(args.output_dir)
    output_file = Path(args.output_dir) / "relationships.jsonl"
    computed = load_computed(output_file)
    llm = build_llm(
        args.model,
        provider=args.provider,
        use_gpt=args.gpt,
        ollama_base_url=args.ollama_base_url,
        api_key=args.api_key,
        openai_base_url=args.openai_base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    print(f"Using {llm['provider']} model: {llm['model']}")
    print(f"Using attribute source: {args.attribute_source}")
    print(f"Loaded {len(top_level_types)} top-level types")
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
                top_level_types,
                args.topk,
                args.skip_ne_filter,
                allow_self_relationship=args.allow_self_relationship,
                type_graph=type_graph,
                use_subtypes=use_subtypes,
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
            if "source_tables" in group:
                relationship_record["source_tables"] = group["source_tables"]
            fout.write(json.dumps(relationship_record, ensure_ascii=False) + "\n")
            fout.flush()
            computed.add(key)

    print(f"Saved relationships to {output_file}")


def default_result_path(result_root, dataset, layer, shot, result_model, filename):
    return str(Path(result_root) / dataset / layer / str(shot) / result_model / filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer GeSI Step 3 relationships with Ollama or OpenAI-compatible LLMs.")
    parser.add_argument("--dataset", type=str, default="Vision")
    parser.add_argument("--dataset_root", type=str, default="E:/Project/CurrentDataset/datasets")
    parser.add_argument("--result_root", type=str, default="E:/Project/CurrentDataset/result/GeSI")
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--result_model", type=str, default="gpt3")
    parser.add_argument("--model", type=str, default="qwen2.5:14b")
    parser.add_argument("--provider", choices=["auto", "ollama", "openai"], default="auto")
    parser.add_argument("--gpt", action="store_true", help="Use the OpenAI-compatible closed-source backend.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434/")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_ctx", type=int, default=8192)
    parser.add_argument("--num_predict", type=int, default=256)
    parser.add_argument(
        "--attribute_source",
        choices=["ground_truth", "attribute_graph"],
        default="ground_truth",
        help="Use column_gt.csv + groundTruth.csv by default; use attribute_graph to read Attribute treeRefine.pkl.",
    )
    parser.add_argument("--column_gt", type=str, default=None)
    parser.add_argument(
        "--gt_type_column",
        type=str,
        default="superclass",
        help="Column in groundTruth.csv that contains the top-level source type.",
    )
    parser.add_argument("--type_graph", type=str, default=None)
    parser.add_argument("--attribute_graph", type=str, default=None)
    parser.add_argument("--skip_ne_filter", action="store_true")
    parser.add_argument("--use_inherited_attribute_columns", action="store_true")
    parser.add_argument("--allow_self_relationship", action="store_true")
    args = parser.parse_args()

    if args.column_gt is None:
        args.column_gt = str(Path(args.dataset_root) / args.dataset / "column_gt.csv")
    if args.attribute_source == "attribute_graph" and args.type_graph is None:
        args.type_graph = default_result_path(
            args.result_root,
            args.dataset,
            "Type",
            args.shot,
            args.result_model,
            "treeRefine.pkl",
        )
    if args.attribute_source == "attribute_graph" and args.attribute_graph is None:
        args.attribute_graph = default_result_path(
            args.result_root,
            args.dataset,
            "Attribute",
            args.shot,
            args.result_model,
            "treeRefine.pkl",
        )
    if args.output_dir is None:
        selected_provider = choose_provider(args.model, args.provider, args.gpt)
        resolved_model_name = resolve_model_name(args.model, selected_provider)
        args.output_dir = str(
            Path(args.result_root)
            / args.dataset
            / "Relationship"
            / str(args.shot)
            / model_name_for_path(resolved_model_name)
        )
    return args


def main():
    args = parse_args()
    infer_relationships(args)


if __name__ == "__main__":
    main()
