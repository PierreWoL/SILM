import os
import random
from pathlib import Path
from GeSI.response import get_model_answer
from Prompt.Examples import example
from absl import app, logging
from openai import OpenAI
from Sampling import Table
from GeSI.Prompt.ColumnTypePrompts import COL_TEMP_FULL, COL_TEMP_TFULL
from GeSI.Extraction import extract_thing_paths
from GeSI.utils.folder import mkdir
import argparse
from tqdm import tqdm
import json
from langchain_ollama import OllamaLLM

parser = argparse.ArgumentParser(description="Please choose the optimal settings for the parameters")
parser.add_argument('--test_dataset', type=str, default='AddedExp/noiseLevel/60_pct', help="Chosen dataset")
parser.add_argument('--LLM', type=str, default='gpt3.5', help="Chosen LLMs")
parser.add_argument('--k_shot', type=int, default=3, help="# of samples to provide.")
parser.add_argument('--seed', type=int, default=0, help="Random seed.")
parser.add_argument('--trial', type=int, default=0, help="trial sequence.")
parser.add_argument('--Prompt', type=int, default=0, help="Prompt choice")
parser.add_argument("--gpt",  type=bool,default=True, help="whether using gpt")

args = parser.parse_args()
llm_mp = {'llama7b': 'Llama-3.1-8B',
          'qwen14b': 'Qwen/Qwen2.5-14B',
          'qwen32b': 'qwen2.5:32b',
          'gpt3.5': 'gpt-3.5-turbo-1106',
          'gpt4': 'gpt-4-1106-preview'}

model_name = 'qwen2.5:14b' # 'qwen2.5:14b' llm_mp[args.LLM]
ollama_base_url = "http://localhost:11434/"
llm_ollama = OllamaLLM(
    base_url=ollama_base_url,
    model= model_name
)
llm_gpt = OpenAI(api_key="")

llm = llm_ollama
if args.gpt is True:
    llm = llm_gpt
    print("using gpt for inferring...")

output_dir = f"result/GeSI/{args.test_dataset}/Attribute/{args.k_shot}/gpt3/" #{args.LLM}/
print(args.test_dataset, args.LLM, args.k_shot, args.seed, args.trial, args.Prompt, output_dir)






def llama_batch_infer(promptsDict: dict, output_path, batch_size=1):

        # with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(promptsDict), batch_size), desc="GPU worker"):
            keys_batch = list(promptsDict.keys())[i:i + batch_size]
            for key in keys_batch:
                cols_tuple = promptsDict[key]
                result_cur_table = {"id": key, "attrs": []}
                for prompt_tuple in cols_tuple:
                    col, prompt = prompt_tuple
                    messages = [
                        {"role": "user",
                         "content": prompt,
                         }
                    ]
                    # this is for individual inference
                    try:
                        decoded = get_model_answer(llm, messages)
                        cleaned = extract_thing_paths(decoded)
                        # print(decoded)
                        result_cur_table["attrs"].append({
                            "column": col,
                            "paths": cleaned
                        })
                    except Exception as e:
                        print(f" {key} column {col} failed: {e}")
                        result_cur_table["attrs"].append({
                            "column": col,
                            "paths": None
                        })
                        continue
                with open(output_path, "a", encoding="utf-8") as f:
                    json_line = json.dumps(result_cur_table, ensure_ascii=False)
                    f.write(json_line + "\n")




def split_evenly_dict(d: dict, n: int) -> list[dict]:
    result = [dict() for _ in range(n)]
    for key, value_list in d.items():
        k, m = divmod(len(value_list), n)
        start = 0
        for i in range(n):
            end = start + k + (1 if i < m else 0)
            result[i][key] = value_list[start:end]
            start = end
    return result


def merge_jsonl_files(input_dir, output_file, prefix="result_gpu"):
    input_dir = os.path.abspath(input_dir)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for fname in sorted(os.listdir(input_dir)):
            if fname.startswith(prefix) and fname.endswith(".jsonl"):
                fpath = os.path.join(input_dir, fname)
                print(f"Merging JSON files from Path: {fpath}")
                with open(fpath, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
    print(f"✅ Merging Complete: {output_file}")


def main(_):
    random.seed(args.seed)
    out_dir = Path(output_dir)
    mkdir(out_dir)
    out_file = out_dir / "results.jsonl"
    # examples = sample_examples(args.k_shot, args.test_dataset)
    examples = example(args.k_shot)

    if args.Prompt == 0:
        promptTemplate = COL_TEMP_FULL
    else:
        promptTemplate = COL_TEMP_TFULL
    # print("Example prompt:\n%s",
    # promptTemplate.render(examples=examples))
    computed = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))
    test_pages = Table.load(args.test_dataset, sample_size=5, summ_stats=True, other_col=False, isText=False)
    # We filter the columns that have been inferred before
    test_pages = [
        {'id': sample["id"], "type": sample["type"], "table": sample["table"]}
        for sample in test_pages
        if sample["id"] not in computed
    ][:]
    # print(test_pages)
    print("Current dataset size that needs to be inferred: ", len(test_pages))

    logging.info("Computing responses for %d pages", len(test_pages))
    prompts_dict = {}
    for sample in test_pages:
        prompts_dict[sample["id"]] = []
        for col in sample["table"].columns:
            select_col = sample["table"][col].dropna().tolist()
            k = min(10, len(select_col) - 1)
            if k <= 1:
                col_content = select_col
            else:
                col_content = random.choices(select_col, k=k)
            if args.Prompt == 0:
                prompt = promptTemplate.render(
                    header=col,
                    col=col_content,
                    type=sample["type"],
                    examples=examples)
            else:
                prompt = promptTemplate.render(
                    header=col,
                    col=col_content,
                    type=sample["type"],
                    table=sample["table"],
                    examples=examples)
            prompts_dict[sample["id"]].append((col, prompt))
    sample_show = list(prompts_dict.keys())[1]
    print(sample_show, prompts_dict[sample_show][1])
    llama_batch_infer(prompts_dict, out_file, batch_size=1)


if __name__ == "__main__":
    app.run(main)
