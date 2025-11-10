import os
import random
from multiprocessing import Process
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from Prompt.Examples import example
import torch
from absl import app, logging
from openai import OpenAI
from Sampling import Table
from Prompt.ColumnTypePrompts import COL_TEMP_FULL, COL_TEMP_TFULL
from utils.folder import mkdir
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import json
from collections import defaultdict

parser = argparse.ArgumentParser(description="Please choose the optimal settings for the parameters")
parser.add_argument('--test_dataset', type=str, default='WDC', help="Chosen dataset")
parser.add_argument('--LLM', type=str, default='qwen14b', help="Chosen LLMs")
parser.add_argument('--k_shot', type=int, default=3, help="# of samples to provide.")
parser.add_argument('--seed', type=int, default=0, help="Random seed.")
parser.add_argument('--trial', type=int, default=0, help="trial sequence.")
parser.add_argument('--Prompt', type=int, default=0, help="Prompt choice")

args = parser.parse_args()

output_dir = f"Result/{args.test_dataset}/Step2/Prompt{args.Prompt}/{args.trial}/{args.LLM}/"
print(args.test_dataset, args.LLM, args.k_shot, args.seed, args.trial, args.Prompt, output_dir)

llm_mp = {'llama7b': 'Llama-3.1-8B',
          'qwen14b': 'Qwen/Qwen2.5-14B',
          'qwen32b': 'qwen2.5:32b',
          'gpt3.5': 'gpt-3.5-turbo-1106',
          'gpt4': 'gpt-4-1106-preview'}


def ask_gpt(id_prompt, prompt_tuples, client):
    result = {"id": id_prompt, "attrs": []}
    for col_name, prompt in prompt_tuples:
        try:
            response = client.chat.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            result["attrs"].append({
                "column": col_name,
                "value": response['choices'][0].message.content
            })

        except Exception as e:
            result["attrs"].append({
                "column": col_name,
                "value": None
            })
            print(id_prompt, "'s column ", col_name, " has error ", str(e))
    return result


def llama_batch_infer(promptsDict: dict, output_path, batch_size=1):
    if 'gpt' in args.LLM:
        llm = OpenAI(api_key="your api key")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ask_gpt, idx, prompt_tuples, llm) for idx, prompt_tuples in promptsDict.items()]
            with open(output_path, "a", encoding="utf-8") as fout:
                for f in tqdm(as_completed(futures), total=len(futures)):
                    result = f.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
    else:
        model_id = llm_mp[args.LLM]
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_skip_modules=None
        )
        acess_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", token=acess_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            device_map="auto",
            token=acess_token
        )
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
                    prompt_token = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # this is for individual inference
                    try:
                        inputs = tokenizer(
                            [prompt_token],
                            return_tensors="pt",
                        ).to(model.device)

                        generated_ids = model.generate(
                            inputs.input_ids,
                            cache_implementation="static",
                            max_new_tokens=256,
                            # do_sample=True,
                            # temperature=0.7,
                            # top_k=50,
                            # top_p=0.95
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in
                            zip(inputs.input_ids, generated_ids)
                        ]

                        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        #print(decoded)
                        result_cur_table["attrs"].append({
                                    "column": col,
                                    "paths": decoded
                                })
                    except Exception as e:
                        print(f" {key} column {col} failed: {e}")
                        torch.cuda.empty_cache()
                        result_cur_table["attrs"].append({
                            "column": col,
                            "paths": None
                        })
                        continue
                with open(output_path, "a", encoding="utf-8") as f:
                    json_line = json.dumps(result_cur_table, ensure_ascii=False)
                    f.write(json_line + "\n")

            """
            keys_batch = list(promptsDict.keys())[i:i + batch_size]
            batch_prompts = []
            ids_batch = []
            for key in keys_batch:
                cols_tuple = promptsDict[key]
                for prompt_tuple in cols_tuple:
                    col, prompt = prompt_tuple

                    messages = [
                        {"role": "user",
                         "content": prompt,
                         }
                    ]
                    prompt_token = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        
                    ids_batch.append(f"{key}.{col}")
                    batch_prompts.append(prompt_token)
            try:
                ### This is batch processing, currently has no additional computational cost for this
                # Tokenize
                inputs = tokenizer(batch_prompts, return_tensors="pt", max_length=4096,
                                   padding=True, truncation=True).to(model.device)
                # Generate
                with torch.no_grad():
                        outputs = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            cache_implementation="static",
                            # max_new_tokens=256,
                            do_sample=False,
                            # temperature=0.7,
                            # top_p=0.95
                        )
    
                # Strip input prompt from outputs
                outputs_trimmed = [
                        output[len(input_each):] for input_each, output in zip(inputs.input_ids, outputs)
                    ]
                decoded = tokenizer.batch_decode(outputs_trimmed, skip_special_tokens=True)
                # Print each result
                # write samples one by one
                grouped = defaultdict(dict)
                for id_, response in zip(ids_batch, decoded):
                        table_name, column_name = id_.rsplit(".", 1)
                        grouped[table_name][column_name] = response
                with open(output_path, "a", encoding="utf-8") as f:
                    for table_id, attrs in grouped.items():
                        result = {"id": table_id, "attrs": []}
                        for col_name, response_col in attrs.items():
                                result["attrs"].append({
                                    "column": col_name,
                                    "value": response_col
                                })
                        json_line = json.dumps(result, ensure_ascii=False)
                        f.write(json_line + "\n")
            except Exception as e:
                print(f" Batch {i} failed: {e}")
                continue    
            """


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
    #print("Example prompt:\n%s",
         # promptTemplate.render(examples=examples))
    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))
    test_pages = Table.load(args.test_dataset, sample_size=5, summ_stats=True, other_col=False)
    # We filter the columns that have been inferred before
    test_pages = [
                     {'id': sample["id"], "type": sample["type"], "table": sample["table"]}
                     for sample in test_pages
                     if sample["id"] not in computed
                 ][:]
    # print(test_pages)
    print("Current dataset size that needs to be inferred: ",len(test_pages))

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
