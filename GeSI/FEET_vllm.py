"""This script is idempotent."""

import json
import os
import random
from pathlib import Path

import torch
from absl import app, flags, logging
from vllm import LLM, SamplingParams
from Sampling.Examples import sample_examples, sampleFEET
from Sampling import Table
from FineGrainedType import PROMPT_TEMPLATE_FULL
from utils import batch, setup_logging, textpbar
from utils.folder import mkdir
import re

FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", "WDC", "Path to the test dataset")
flags.DEFINE_integer("k_shot", 3, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")
# flags.DEFINE_integer("samplings", 3, "Sampling strategy.")
sampling = 3
mode = 5
modelN = "deepseek"
#flags.DEFINE_string("output_dir", "Result/GDS/Prompt0/3/deepseek/s1/", "Path to the output directory")

from huggingface_hub import snapshot_download


from utils.Utils import extract_answer_regex


local_model_path = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")


def main(_):
    random.seed(FLAGS.seed)
    if mode == 3:
        fold_name = "filter"
    elif mode ==4:
        fold_name = "cutoff"
    elif mode == 5:
        fold_name = "cutoff_filter"
    output_dir = f"Result/{FLAGS.test_dataset}/Prompt0/{FLAGS.k_shot}/fet/{modelN}/{fold_name}/"
    out_dir = Path(output_dir)
    mkdir(out_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "FEET_results.jsonl"
    examples = sampleFEET(FLAGS.k_shot, FLAGS.test_dataset)
    print(len(examples))
    logging.info(
        "Example prompt:\n%s",
        PROMPT_TEMPLATE_FULL.render(examples=examples),
    )

    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))

    # test_pages = Table.load(FLAGS.test_dataset, sample_s=sampling,sample_size=8, summ_stats=True, other_col=False)
    test_pages = Table.load(FLAGS.test_dataset, sample_size=8, sample_s=mode, summ_stats=True, other_col=False)

    print(len(test_pages),test_pages[0])
    test_pages = [
        {'id': sample["id"], "table": sample["table"]}
        for sample in test_pages
        if sample["id"] not in computed
    ]
    # print(test_pages)
    logging.info("Computing responses for %d pages", len(test_pages))
    llm = LLM(
        model=local_model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        # max_num_batched_tokens=8192,
        max_model_len=8192,
        max_seq_len_to_capture=8192,
        # block_size=64,
        enable_chunked_prefill=False,
    )
    tokenizer = llm.get_tokenizer()
    pbar = textpbar(len(test_pages))

    for pages in batch(test_pages, 5):
        prompts = []
        for page in pages:
            messages = [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE_FULL.render(
                        table=page["table"],
                        examples=examples,
                    ),
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        print(prompts[0])
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=2048,
                # stop=["</think>\n\n"],
                seed=42,  # FLAGS.seed,
            ),
        )
        for page, out in zip(pages, outputs):
            answer = extract_answer_regex(out.outputs[0].text)
            print(page["id"], answer)

            with open(out_file, "a") as f:
                item = {
                    # **page,
                    "id": page["id"],
                    "type": answer,
                }
                f.write(json.dumps(item) + "\n")
            pbar.update()


if __name__ == "__main__":
    app.run(main)
