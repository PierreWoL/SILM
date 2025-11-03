"""This script is idempotent."""
import json
import random
from pathlib import Path
import torch
from absl import app, flags, logging
import re
from Sampling.Examples import sample_examples
from Sampling import Table
from PROMPTS import PROMPT_TEMPLATE_FULL, \
    PROMPT_TEMPLATE_CONSTRAINT_FULL, PROMPT_TEMPLATE_CONSTRAINTSET_FULL, \
    PROMPT_TEMPLATE_CONSTRAINTSTL_FULL, PROMPT_TEMPLATEABS_FULL
from response import get_model_answer
from utils import batch, setup_logging, textpbar
from huggingface_hub import snapshot_download
from utils.folder import mkdir
from models import chooseLLM




FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", "GoogleSearch", "Path to the test dataset")
flags.DEFINE_string("output_dir", "Result/GoogleSearch/Prompt0/qwen14/", "Path to the output directory")
flags.DEFINE_integer("k_shot", 3, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_bool("constraintFET", False, "constraint for the leaf type.")
flags.DEFINE_bool("constraintSLT", False, "constraint for the second level.")
flags.DEFINE_bool("constraintABS", True, "constraint for the second level not using abstract type.")

abs_example = ['entity', 'something', 'unidentified entity', 'conceptual entity', 'certain thing', 'former entity',
               'named entity',
               'result', 'source', 'substance', 'group', 'object', 'information', 'part']

model_name = 'qwen:32b'
mode = 5
llm = chooseLLM(model_name)
if mode == 3:
    fold_name = "filter"
elif mode == 4:
    fold_name = "cutoff"
elif mode == 5:
    fold_name = "cutoff_filter"

model_folder = model_name.split(":")[0]


def extract_answer_regex(text):
    match = re.split(r"</think>\s*\n\n", text, flags=re.IGNORECASE)
    return match[1] if len(match) > 1 else text


def main(_):
    random.seed(FLAGS.seed)
    output_dir = f"Result/{FLAGS.test_dataset}/Prompt0/{FLAGS.k_shot}/{model_folder}/"
    out_dir = Path(output_dir)
    mkdir(out_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    examples = sample_examples(FLAGS.k_shot, FLAGS.test_dataset)
    print(len(examples))
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
    test_pages = Table.load(FLAGS.test_dataset, sample_size=5, sample_s=mode, summ_stats=True, other_col=False)

    print(test_pages[0], len(test_pages))

    if FLAGS.constraintFET is True:
        fet_data = {}
        fet_file = f"Result/{FLAGS.test_dataset}/Prompt0/3/fet/{model_folder}/{fold_name}/FEET_results.jsonl"
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
    logging.info("Computing responses for %d pages", len(test_pages))

    for page in test_pages:
        if FLAGS.constraintFET is True:
            content = template.render(table=page["table"], fet=page["fet"], examples=examples)
            # print(content)
        else:
            if FLAGS.constraintABS is True:
                content = template.render(table=page["table"], examples=examples, abs_list=abs_example)
            else:
                content = template.render(table=page["table"], examples=examples)
        messages = [{"role": "user", "content": content}]
        answer = get_model_answer(llm, messages)
        print(page["id"], answer)
        with open(out_file, "a") as f:
            # answer = extract_answer_regex(FEET.outputs[0].text)

            item = {
                "id": page["id"],
                "hierarchy": answer,
            }
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)
