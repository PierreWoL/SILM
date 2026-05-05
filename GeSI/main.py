"""This script is idempotent."""

import json
import os
import random
from pathlib import Path
from absl import app, flags, logging
from openai import OpenAI

from GeSI.Extraction import extract_thing_paths
from GeSI.Sampling.Examples import sample_examples
from GeSI.Sampling import Table
from GeSI.PROMPTS import PROMPT_TEMPLATE_FULL, PROMPT_TEMPLATEABS_FULL, PROMPT_TEMPLATE_CONSTRAINTSET_FULL, \
    PROMPT_TEMPLATE_CONSTRAINTSTL_FULL, PROMPT_TEMPLATE_CONSTRAINT_FULL
from response import get_model_answer
from langchain_ollama import OllamaLLM
from GeSI.utils.folder import mkdir

FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", "AddedExp/noiseLevel/20_pct", "Path to the test dataset")
flags.DEFINE_string("output_dir", "result/GeSI/AddedExp/noiseLevel/20_pct/Type/3/qwen14/", "Path to the output directory")
flags.DEFINE_integer("k_shot", 3, "Number of samples to provide.")
flags.DEFINE_integer("MaxiDepth", 5, "Number of samples to provide.")

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_bool("constraintFET", False, "constraint for the leaf type.")
flags.DEFINE_bool("constraintSLT", False, "constraint for the second level.")
flags.DEFINE_bool("constraintABS", True, "constraint for the second level not using abstract type.")
flags.DEFINE_bool("gpt", False, "whether using gpt")

abs_example = ['entity', 'something', 'unidentified entity', 'conceptual entity', 'certain thing', 'former entity',
               'named entity',
               'result', 'source', 'substance', 'group', 'object', 'information', 'part']
model_name = 'qwen2.5:14b'
ollama_base_url = "http://localhost:11434/"
llm_ollama = OllamaLLM(
    base_url=ollama_base_url,
    model= model_name,
options={
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 8192,
        "stop": ["\n\n"]
    }
)


llm_gpt = OpenAI(api_key="")


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    mkdir(out_dir)
    examples = sample_examples(FLAGS.k_shot,FLAGS.test_dataset)
    print(len(examples))
    llm= llm_ollama
    if FLAGS.gpt is True:
        llm = llm_gpt
        print("using gpt for inferring...")
    if FLAGS.constraintFET is False and FLAGS.constraintSLT is False:
        if FLAGS.constraintABS is True:
            template = PROMPT_TEMPLATEABS_FULL
            out_file = out_dir / "results_constraintABS.jsonl"
        else:
            template = PROMPT_TEMPLATE_FULL
            out_file = out_dir / "results.jsonl"

    elif FLAGS.constraintFET is True and FLAGS.constraintSLT is False:
        template = PROMPT_TEMPLATE_CONSTRAINTSET_FULL
        out_file = out_dir / "results_constraintFET.jsonl"

    elif FLAGS.constraintFET is False and FLAGS.constraintSLT is True:
        template = PROMPT_TEMPLATE_CONSTRAINTSTL_FULL
        out_file = out_dir / "results_constraintSTL.jsonl"

    else:
        template = PROMPT_TEMPLATE_CONSTRAINT_FULL
        out_file = out_dir / "results_constraintFull.jsonl"

    if FLAGS.constraintABS is True:
        logging.info(
            "Example prompt:\n%s",
            template.render(
                Mdep =FLAGS.MaxiDepth,
                abs_list=abs_example,
                examples=examples,

            ),
        )
    else:
        logging.info(
            "Example prompt:\n%s",
            template.render(
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
    if FLAGS.constraintFET is True:
        fet_data = {}
        fet_file = out_dir / "FEET_results.jsonl"
        with open(fet_file, "r") as f:
            for line in f:
                data = json.loads(line)
                fet_data[data['id']] = data['type']
        test_pages = [
            {'id': sample["id"], "table": sample["table"], 'fet': fet_data[sample["id"]]}
            for sample in test_pages
            if sample["id"] not in computed
        ]
    else:
        test_pages = [
            {'id': sample["id"], "table": sample["table"]}
            for sample in test_pages
            if sample["id"] not in computed
        ]
    #print(test_pages)
    logging.info("Computing responses for %d pages", len(test_pages))

    for table in test_pages:
        if FLAGS.constraintFET is True:
            content = template.render(Mdep =FLAGS.MaxiDepth ,table=table["table"], fet=table["fet"], examples=examples)
        else:
            if FLAGS.constraintABS is True:
                content = template.render(Mdep =FLAGS.MaxiDepth ,table=table["table"], examples=examples, abs_list=abs_example)
            else:
                content = template.render(Mdep =FLAGS.MaxiDepth ,table=table["table"], examples=examples)
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        """
        hierarchy = get_model_answer(llm, messages)
        cleaned = extract_thing_paths(hierarchy)
        if cleaned == "":
            print("No valid Thing paths found.")
        else:
            print(table["id"], cleaned) 
        """
        cleaned = ""
        for attempt in range(3):
            hierarchy = get_model_answer(llm, messages)
            cleaned = extract_thing_paths(hierarchy)
            if cleaned != "":
                result = {"id": table["id"], "hierarchy": cleaned}
                print(table["id"],cleaned)
                with open(os.path.join(FLAGS.output_dir, "results.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                break
            print(f"No valid Thing paths found. Retry {attempt + 1}/3")
        if cleaned == "":
            print(f"Skip table {table['id']}: no valid Thing paths after 3 retries.")
            continue


if __name__ == "__main__":
    app.run(main)
