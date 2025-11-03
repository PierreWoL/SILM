import json
import os
import random
from pathlib import Path
from absl import app, flags, logging
#from vllm import LLM, SamplingParams
from utils import batch, setup_logging, textpbar
from Prompt.Examples import example
from Prompt.Step2 import ATTRPROMPT_TEMPLATESPL_FULL
from Sampling import Table
from response import get_model_answer
from utils.folder import mkdir
from utils.Utils import extract_answer_regex, chooseLLMs



FLAGS = flags.FLAGS
mode = 0
dataset = "WDC"

flags.DEFINE_string("output_dir", f"Result/{dataset}/Step2/TA/split/1/GPT4/", "Path to the output directory")
flags.DEFINE_integer("k_shot", 1, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")


# model_name = "qwen2.5:14b"
#


model_name = 'gpt'


def AttributePer():
    output_file = "TAresults.jsonl"
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    mkdir(out_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / output_file
    examples = example(FLAGS.k_shot)
    logging.info(
        "Example prompt:\n%s",
        ATTRPROMPT_TEMPLATESPL_FULL.render(
            examples=examples,
        ),
    )
    computed = set()
    if out_file.exists():
        with open(out_file, "r", encoding='utf-8') as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))
    test_pages = Table.load(dataset, sample_size=5, summ_stats=True, other_col=False)
    # print(test_pages[0])
    test_pages = [
        {'id': sample["id"], "table": sample["table"], 'type': sample["type"]}
        for sample in test_pages
        if sample["id"] not in computed
    ]
    llm = chooseLLMs(model_name, VLLM=False)
    """
   
    if isinstance(llm, LLM):
        tokenizer = llm.get_tokenizer()
        pbar = textpbar(len(test_pages))
        for pages in batch(test_pages, 5):
            prompts = []
            for page in pages:
                for col in page["table"].columns:
                    select_col = page["table"][col].dropna().tolist()
                    k = min(10, len(select_col) - 1)
                    if k <= 0:
                        col_content = select_col
                    else:
                        col_content = random.choices(select_col, k=k)
                    prompt = ATTRPROMPT_TEMPLATESPL_FULL.render(
                        header=col,
                        table=page["table"],
                        col=col_content,
                        type=page["type"],
                        examples=examples)
                    # print(prompt)
                    messages = [{"role": "user", "content": prompt}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)
            print(prompts[0])
            outputs_columns = llm.generate(
                prompts,
                sampling_params=SamplingParams(
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=2048,
                    # stop=["</think>\n\n"],
                    seed=42,  # FLAGS.seed,
                ),
            )
            last_index = 0
            for page in pages:
                answers = []
                cols = page["table"].columns
                output_attrs = outputs_columns[last_index: last_index+len(cols)]
                for output_attr in output_attrs:
                    answer = extract_answer_regex(output_attr.outputs[0].text)
                    answers.append(answer)
                print(cols,answers)
                with open(out_file, "a") as f:
                    item = {
                        # **page,
                        "id": page["id"],
                        "type": answers,
                    }
                    f.write(json.dumps(item) + "\n")
                last_index += len(cols)
                pbar.update()
    else:
     """
    logging.info("Computing responses for %d pages", len(test_pages))
    for table in test_pages:
        attrs = []
        for col in table["table"].columns:
            select_col = table["table"][col].dropna().tolist()
            k = min(10, len(select_col) - 1)
            if k <= 0:
                col_content = select_col
            else:
                col_content = random.choices(select_col, k=k)
            prompt = ATTRPROMPT_TEMPLATESPL_FULL.render(
                header=col,
                table=table["table"],
                col=col_content,
                type=table["type"],
                examples=examples)
            # print(prompt)
            messages = [{"role": "user", "content": prompt}]
            attribute_name = get_model_answer(llm, messages)
            attrs.append(attribute_name)
        result = {"id": table["id"], "attrs": attrs}
        print(table["id"], attrs)
        with open(os.path.join(FLAGS.output_dir, output_file), "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(_):
    # if mode ==1 :
    # allAttribute()
    # else:
    print("start attribute per inference")
    AttributePer()


if __name__ == "__main__":
    app.run(main)
