"""This script is idempotent."""

import json
import os
import random
from pathlib import Path

import torch
from absl import app, flags, logging
from openai import OpenAI

from Sampling.Examples import sample_examples
from Sampling import Table
from PROMPTS import PROMPT_TEMPLATE_FULL
from response import get_model_answer
from utils import batch, setup_logging, textpbar
from langchain_ollama import OllamaLLM

from utils.folder import mkdir

FLAGS = flags.FLAGS

flags.DEFINE_string("test_dataset", "WDC", "Path to the test dataset")
flags.DEFINE_string("output_dir", "Result/WDC/Prompt0/1/qwen/", "Path to the output directory")
flags.DEFINE_integer("k_shot", 3, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")
model_name = 'qwen2.5:14b'
ollama_base_url = "http://localhost:11434/"
llm = OllamaLLM(
    base_url=ollama_base_url,
    model= model_name,
options={
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 8192,
        "stop": ["\n\n"]
    }
)


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    mkdir(out_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "results.jsonl"
    examples = sample_examples(FLAGS.k_shot,FLAGS.test_dataset)
    print(len(examples))
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

    #with open(FLAGS.test_dataset, "r") as f:
        #test_pages = [json.loads(line) for line in f.readlines()]

    test_pages = Table.load(FLAGS.test_dataset, sample_size=5, summ_stats=True, other_col=False)
    print(test_pages[0])
    test_pages = [
            {'id': sample["id"], "table": sample["table"]}
            for sample in test_pages
            if sample["id"] not in computed
        ]
    #print(test_pages)
    logging.info("Computing responses for %d pages", len(test_pages))

    for table in test_pages:
        prompt = PROMPT_TEMPLATE_FULL.render(
            table=table["table"],
            examples=examples)
        messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        hierarchy = get_model_answer(llm, messages)
        print(table["id"], hierarchy)
        result={"id": table["id"], "hierarchy": hierarchy}
        with open(os.path.join(FLAGS.output_dir,"results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    app.run(main)
