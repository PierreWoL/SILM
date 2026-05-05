import random
import tiktoken
from transformers import AutoTokenizer
from GeSI.PROMPTS import PROMPT_TEMPLATE_FULL as template
from GeSI.Prompt.ColumnTypePrompts import COL_TEMP_FULL as promptTemplate
from absl import app, flags, logging
from GeSI.Sampling.Examples import sample_examples
from GeSI.Sampling import Table



FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", "OD_Large", "Path to the test dataset")
flags.DEFINE_integer("k_shot", 3, "Number of samples to provide.")
flags.DEFINE_integer("MaxiDepth", 5, "Number of samples to provide.")


def count_openai_message_tokens(messages, model="gpt-3.5-turbo-1106"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message

        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))

            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens



def count_qwen_message_tokens(
    messages,
    model_name="Qwen/Qwen2.5-14B-Instruct"
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    return len(token_ids)



def main(_):
    examples = sample_examples(FLAGS.k_shot,FLAGS.test_dataset)
    test_pages = Table.load(FLAGS.test_dataset, sample_size=5, summ_stats=True, other_col=False)
    test_pages = [
                {'id': sample["id"], "table": sample["table"]}
                for sample in test_pages
            ]
    total_tokens_qwen = 0
    total_tokens_gpt = 0
    for table in test_pages:
        content = template.render(Mdep=FLAGS.MaxiDepth, table=table["table"], examples=examples)
        messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        openai_tokens = count_openai_message_tokens(messages, model="text-embedding-3-large")
        qwen_tokens = count_qwen_message_tokens(messages, model_name="Qwen/Qwen2.5-14B-Instruct")
        total_tokens_qwen += qwen_tokens
        total_tokens_gpt += openai_tokens
        print(table["id"], total_tokens_qwen, total_tokens_gpt)
    print("OpenAI token count:", total_tokens_gpt)
    print("Qwen token count:", total_tokens_qwen)

    if FLAGS.test_dataset != "OD_Large":
        total_tokens_qwen_col = 0
        total_tokens_gpt_col = 0
        testCol_pages = Table.load(FLAGS.test_dataset, sample_size=5, summ_stats=True, other_col=False, isText=False)
        testCol_pages = [{'id': sample["id"], "type": sample["type"], "table": sample["table"]} for sample in testCol_pages]
        for sample in testCol_pages:
            for col in sample["table"].columns:
                select_col = sample["table"][col].dropna().tolist()
                k = min(10, len(select_col) - 1)
                if k <= 1:
                    col_content = select_col
                else:
                    col_content = random.choices(select_col, k=k)
                prompt = promptTemplate.render(
                        header=col,
                        col=col_content,
                        type=sample["type"],
                        examples=examples)
                messages = [
                    {"role": "user",
                     "content": prompt,
                     }
                ]
                openai_col_tokens = count_openai_message_tokens(messages, model="text-embedding-3-large")
                qwen_col_tokens = count_qwen_message_tokens(messages, model_name="Qwen/Qwen2.5-14B-Instruct")
                total_tokens_qwen_col += qwen_col_tokens
                total_tokens_gpt_col += openai_col_tokens
            print(sample["id"], total_tokens_qwen_col, total_tokens_gpt_col)

        print("OpenAI token count for attributes:", total_tokens_gpt)
        print("Qwen token count for attributes:", total_tokens_qwen)

if __name__ == "__main__":
    app.run(main)