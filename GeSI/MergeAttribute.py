import json
import math
import pickle
import random
from pathlib import Path
from absl import app, logging
from langchain_ollama import OllamaLLM
from openai import OpenAI
from Prompt.Step2 import MERGE_PROMPT
from response import get_model_answer
from utils import setup_logging
from utils.folder import mkdir
import ast
from AttributeResolution import fill_missing_attributes
from traverseGraph import build_complete_bottom_up_levels


# deepseek-r1:8b


def mergeAttribute(G, currentNode, llm_model, proportion=0.3):
    def filter_dict_by_value_length(d, n):
        return {k: v for k, v in d.items() if isinstance(v, (list, set, tuple, str)) and len(v) >= n}

    children_nodes = list(G.successors(currentNode))
    keep_length = math.floor(len(children_nodes) * proportion)
    aggre_attrs = []
    for child in children_nodes:
        AR_CUR = G.nodes[child].get('AR', [])
        if AR_CUR is None:
            AR_CUR = []
        aggre_attrs.extend(AR_CUR)
    print(len(aggre_attrs), aggre_attrs)
    prompt = MERGE_PROMPT.render(
        type=currentNode,
        attrs=aggre_attrs)
    # print(prompt)
    messages = [{"role": "user", "content": prompt, }]
    groups_attr = get_model_answer(llm_model, messages)
    try:
        real_dict = ast.literal_eval(groups_attr)
        for attempt in range(3):
            real_dict = fill_missing_attributes(aggre_attrs, real_dict)
            if real_dict != -1:
                G.nodes[currentNode]['RA_dict'] = real_dict
                kept_attrs = list(filter_dict_by_value_length(real_dict, keep_length).keys())
                G.nodes[currentNode]['AR'] = G.nodes[currentNode]['AR'].extend(kept_attrs)
                G.nodes[currentNode]['visited'] = 'True'
                break
            else:
                print(f"Retry attempt {attempt + 1} for test_page: {currentNode}")
    except Exception as e:
        print(e, "\n", currentNode, " failed!")


def InferAttribute(dataset, output_dir, llm_model, seed):
    output_file = "MAresults.pkl"
    random.seed(seed)
    out_dir = Path(output_dir)
    mkdir(out_dir)
    setup_logging(out_dir, "main")
    out_file = out_dir / output_file
    computed = set()
    with open(f'datasets/{dataset}/graphGroundTruth.pkl', 'rb') as f:
        G = pickle.load(f)
    if out_file.exists():
        with open(out_file, 'rb') as f:
            G = pickle.load(f)
    logging.info("Loaded %d computed pages", len(computed))
    with open(f"Result/{dataset}/Step2/AR/1/qwen/ARresults.jsonl", "r", encoding='utf-8') as f:
        all = [json.loads(line) for line in f]
    specificClass = {i["class"]: list(i["inferred"].keys()) for i in all}
    for node in G.nodes:
        AR_current = G.nodes[node].get('AR', None)
        visited_current = G.nodes[node].get('visited', None)
        if visited_current is None:
            G.nodes[node]['visited'] = False
        if AR_current is None:
            G.nodes[node]['AR'] = []
            if node in specificClass:
                G.nodes[node]['AR'] = specificClass[node]

    top_level_types = list(G.successors("Thing"))
    # print("Top-Level Types:", top_level_types)
    for top_node in top_level_types:
        levels = build_complete_bottom_up_levels(G, top_node=top_node)
        for i, level in enumerate(levels):
            # print(f"Level{i} (from lowest level to) {i} level, {len(levels)} level) in total:")
            for node in level:
                if i != 0:
                    children = list(G.successors(node))
                    print(f"The {node} has children: {children}")
                    if G.nodes[node]['visited'] is False:
                        mergeAttribute(G, node, llm_model)
                        with open(out_file, 'wb') as f:
                            pickle.dump(G, f)


def main(_):
    dataset = "GDS"
    key = "key"
    client = OpenAI(api_key=key)
    output_dir = f"Result/{dataset}/Step2/MA/1/qwen/"
    seed = 0
    model_name = 'qwen2.5:14b'
    ollama_base_url = "http://localhost:11434/"
    llm = OllamaLLM(
        base_url=ollama_base_url,
        model=model_name,
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 8192,
            "stop": ["\n\n"]
        }
    )
    InferAttribute(dataset, output_dir, llm, seed)


if __name__ == "__main__":
    app.run(main)
