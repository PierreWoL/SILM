import json
import os
import random
from pathlib import Path
from absl import app, logging
from Prompt.Step2 import ARPROMPT_TEMPLATE_FULL, REST_PROMPT
from Sampling import Table
from response import get_model_answer
from utils.Utils import chooseLLMs, extract_answer_regex
from utils.folder import mkdir
import ast
# from vllm import SamplingParams
from utils import batch, setup_logging, textpbar

# deepseek-r1:8b
mode = 0
dataset = "GDS"
filter_num = 0
model_name = 'qwen2.5:14b'
folder = "qwen"
output_dir = f"Result/{dataset}/Step2/AR/1/{folder}/filter{filter_num}/"
TA_path = f"Result/{dataset}/Step2/TA/split/1/{folder}/TAresults.jsonl"
k_shot = 1
seed = 0



def fill_missing_attributes(origin, current: dict, max_stagnant_rounds=3, vllm=False):
    llm = chooseLLMs(model_name, VLLM=vllm)
    stagnant_rounds = 0
    prev_missing_count = None
    while True:
        current = {k: [data for data in v if data in origin] for k, v in current.items()}
        grouped_attrs = [v for group in current.values() for v in group]
        missing_attrs = list(set(origin) - set(grouped_attrs))
        missing_attrs = [s for s in missing_attrs if len(s) <= 100]
        print(f" Total: {len(origin)} | Grouped: {len(grouped_attrs)} | Missing: {len(missing_attrs)}")
        if not missing_attrs:
            print("All attributes grouped.")
            break
        if prev_missing_count == len(missing_attrs):
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
        prev_missing_count = len(missing_attrs)
        if stagnant_rounds >= max_stagnant_rounds:
            print(f" Stopped after {stagnant_rounds} stagnant rounds. Remaining ungrouped attributes:")
            print(missing_attrs)
            break
        prompt = REST_PROMPT.render(
            keys=list(current.keys()),
            missing=missing_attrs
        )
        messages = [{"role": "user", "content": prompt}]
        groups_attrs_str = get_model_answer(llm, messages)
        try:
            attr_dict = ast.literal_eval(groups_attrs_str)
        except Exception as e:
            print(" Failed to parse response:", groups_attrs_str)
            raise e
            # return -1
        for k, v in attr_dict.items():
            if k in current:
                current[k].extend(v)
            else:
                current[k] = v
        for k in current:
            current[k] = list(set(current[k]))
    return current


def allAttribute(vllm=False, isFilling=False, filter=0):
    llm = chooseLLMs(model_name, VLLM=vllm)
    output_file = "ARresults.jsonl"
    random.seed(seed)
    out_dir = Path(output_dir)
    mkdir(out_dir)
    out_file = out_dir / output_file
    computed = set()
    if out_file.exists():
        with open(out_file, "r", encoding='utf-8') as f:
            computed.update({json.loads(line)["class"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))
    # test_pages = Table.loadTA(dataset, f"Result/{dataset}/Step2/TA/split/1/gpt3/TAresults.jsonl")
    test_pages = Table.loadTAFilter(dataset, TA_path,
                                    filter=filter)
    test_pages = [{'fet': fet, "attrs": list(attri_list.keys())}
                  for fet, attri_list in test_pages.items()
                  if fet not in computed]
    logging.info("Computing responses for %d pages", len(test_pages))
    if vllm is False:
        for test_page in test_pages:
            # if test_page["fet"] not in ["cricketer Player", "Football Player"]:
            prompt = ARPROMPT_TEMPLATE_FULL.render(
                ea="Suppliers, Supplier name, Companies who supplies, manufacturer, "
                   "Consumer Company, Company who buys the product",
                eao={"Supplier Company": ["Suppliers", "Supplier name", "Companies who supplies"],
                     "Manufacturer Company": ["manufacturer"],
                     "Consumer Company": ["Consumer Company", "Company who buys the product"]},
                attrs=test_page["attrs"])
            messages = [{"role": "user",
                         "content": prompt, }]
            groups_attr = get_model_answer(llm, messages)
            result = None
            try:
                real_dict = ast.literal_eval(groups_attr)
                if isFilling is False:
                    result = {
                        "class": test_page["fet"],
                        "attrs": test_page["attrs"],
                        "inferred": real_dict
                    }
                    print(test_page["fet"], real_dict)
                    with open(os.path.join(output_dir, output_file), "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    for attempt in range(3):
                        real_dict = fill_missing_attributes(test_page["attrs"], real_dict)
                        if real_dict != -1:
                            result = {
                                "class": test_page["fet"],
                                "attrs": test_page["attrs"],
                                "inferred": real_dict
                            }
                            print(test_page["fet"], real_dict)
                            break
                        else:
                            print(f"🔁 Retry attempt {attempt + 1} for test_page: {test_page['fet']}")
                    if result is not None:
                        with open(os.path.join(output_dir, output_file), "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        print(f"❌ Failed to group attributes for {test_page['fet']} after 3 attempts.")
            except Exception as e:
                print(test_page["fet"], " failed!")
    else:
        print("TBC")
        '''
        tokenizer = llm.get_tokenizer()
        pbar = textpbar(len(test_pages))
        for pages in batch(test_pages, 5):
            prompts = []
            for page in pages:
                prompt = ARPROMPT_TEMPLATE_FULL.render(
                    ea="Suppliers, Supplier name, Companies who supplies, manufacturer, "
                       "Consumer Company, Company who buys the product",
                    eao={"Supplier Company": ["Suppliers", "Supplier name", "Companies who supplies"],
                         "Manufacturer Company": ["manufacturer"],
                         "Consumer Company": ["Consumer Company", "Company who buys the product"]},
                    attrs=page["attrs"])
                messages = [{"role": "user",
                             "content": prompt, }]

                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
            print(prompts[0])
            outputs = llm.generate(
                prompts,
                sampling_params=SamplingParams(temperature=0.1,
                                               top_p=0.9, max_tokens=2048, seed=42),
            )
            for page, out in zip(pages, outputs):
                answer = extract_answer_regex(out.outputs[0].text)
                print(page["id"], answer)
                real_dict = ast.literal_eval(answer)
                result = {
                    "class": page["fet"],
                    "attrs": page["attrs"],
                    "inferred": real_dict
                }
                print(page["fet"], real_dict)
                with open(os.path.join(output_dir, output_file), "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            pbar.update()
        '''


def main(_):
    allAttribute(vllm=False, isFilling=False, filter=filter_num)


if __name__ == "__main__":
    app.run(main)
