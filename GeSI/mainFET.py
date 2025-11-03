"""This script is idempotent."""

import json
import os
import random
from pathlib import Path
from absl import app, flags, logging
from Sampling.Examples import sampleFEET
from Sampling import Table
from FineGrainedType import PROMPT_TEMPLATE_FULL
from response import get_model_answer
from utils import setup_logging
#from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.folder import mkdir
from models import chooseLLM
FLAGS = flags.FLAGS
# deepseek-r1:8b
mode = 5
flags.DEFINE_string("test_dataset", "WDC", "Path to the test dataset")
flags.DEFINE_integer("k_shot", 3, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")



model_name = "deepseek-r1:14b"#'qwen:14b'
llm = chooseLLM(model_name)
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
'''

def main(_):
    random.seed(FLAGS.seed)
    if mode == 3:
        fold_name = "filter"
    elif mode ==4:
        fold_name = "cutoff"
    elif mode == 5:
        fold_name = "cutoff_filter"

    model_folder = model_name.split(":")[0]
    output_dir = f"Result/{FLAGS.test_dataset}/Prompt0/{FLAGS.k_shot}/fet/{model_folder}/{fold_name}/"
    out_dir = Path(output_dir)
    mkdir(out_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "FEET_results.jsonl"
    examples = sampleFEET(FLAGS.k_shot, FLAGS.test_dataset)
    logging.info(
        "Example prompt:\n%s",
        PROMPT_TEMPLATE_FULL.render(
            examples=examples,
        ),
    )
    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))
    test_pages = Table.load(FLAGS.test_dataset, sample_size=5, sample_s=mode, summ_stats=True, other_col=False)
    print(test_pages[0])
    test_pages = [
        {'id': sample["id"], "table": sample["table"]}
        for sample in test_pages
        if sample["id"] not in computed
    ]
    logging.info("Computing responses for %d pages", len(test_pages))
    for table in test_pages:
        prompt = PROMPT_TEMPLATE_FULL.render(
            table=table["table"],
            examples=examples)
        #print(prompt)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        FEET = get_model_answer(llm, messages)
        # 使用 chat 模板格式化输入
        '''
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs.input_ids, max_length=128)
        FEET = tokenizer.decode(outputs[0], skip_special_tokens=True)
        '''
        print(table["id"], FEET)
        result = {"id": table["id"], "type": FEET}
        with open(os.path.join(output_dir, "FEET_results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app.run(main)
