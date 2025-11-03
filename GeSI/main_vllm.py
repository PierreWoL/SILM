"""This script is idempotent."""

import json
import os
import random
from pathlib import Path

import torch
from absl import app, flags, logging
import re
from vllm import LLM, SamplingParams
from Sampling.Examples import sample_examples
from Sampling import Table
from PROMPTS import PROMPT_TEMPLATE_FULL, \
    PROMPT_TEMPLATE_CONSTRAINT_FULL, PROMPT_TEMPLATE_CONSTRAINTSET_FULL, \
    PROMPT_TEMPLATE_CONSTRAINTSTL_FULL, PROMPT_TEMPLATEABS_FULL
from utils import batch, setup_logging, textpbar
from huggingface_hub import snapshot_download
from utils.folder import mkdir

FLAGS = flags.FLAGS

flags.DEFINE_string("test_dataset", "GDS", "Path to the test dataset")
flags.DEFINE_string("output_dir", "Result/GDS/Prompt0/0/deepseek/", "Path to the output directory")
flags.DEFINE_integer("k_shot", 0, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_bool("constraintFET", False, "constraint for the leaf type.")
flags.DEFINE_bool("constraintSLT", False, "constraint for the second level.")
flags.DEFINE_bool("constraintABS", True, "constraint for the second level not using abstract type.")

abs_example = ['entity', 'something', 'unidentified entity', 'conceptual entity', 'certain thing', 'former entity',
               'named entity',
               'result', 'source', 'substance', 'group', 'object', 'information', 'part']


def extract_answer_regex(text):
    match = re.split(r"</think>\s*\n\n", text, flags=re.IGNORECASE)
    return match[1] if len(match) > 1 else text


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
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
    test_pages = Table.load(FLAGS.test_dataset, sample_size=5, summ_stats=True, other_col=False)
    print(test_pages[0], len(test_pages))

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
    # print(test_pages)

    logging.info("Computing responses for %d pages", len(test_pages))
    # local_model_path = snapshot_download("andylolu24/ollm-wikipedia")
    local_model_path = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    llm = LLM(
        model=local_model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        # max_num_batched_tokens=8192,
        max_model_len=8192,
        max_seq_len_to_capture=8192,
        max_num_seqs=512,
        # block_size=64,
        enable_chunked_prefill=False,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=2048,
        # stop=None,
        seed=42,
    )
    tokenizer = llm.get_tokenizer()
    pbar = textpbar(len(test_pages))

    for pages in batch(test_pages, 5):
        prompts = []
        for page in pages:
            if FLAGS.constraintFET is True:
                content = template.render(table=page["table"], fet=page["fet"], examples=examples)
            else:
                if FLAGS.constraintABS is True:
                    content = template.render(table=page["table"], examples=examples, abs_list=abs_example)
                else:
                    content = template.render(table=page["table"], examples=examples)
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
        )

        for page, out in zip(pages, outputs):
            with open(out_file, "a") as f:
                answer = extract_answer_regex(out.outputs[0].text)
                print(page["id"], answer)
                item = {
                    "id": page["id"],
                    "hierarchy": answer,
                }
                f.write(json.dumps(item) + "\n")
            pbar.update()


if __name__ == "__main__":
    app.run(main)
