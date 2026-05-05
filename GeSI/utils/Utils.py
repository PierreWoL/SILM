import re
from huggingface_hub import snapshot_download
from langchain_ollama import OllamaLLM
from openai import OpenAI
# from vllm import LLM
import torch

def extract_answer_regex(text):
    match = re.split(r"</think>\s*\n\n", text, flags=re.IGNORECASE)
    return match[1] if len(match) > 1 else text

def chooseLLMs(modelName, VLLM=False):
    if "gpt" in modelName:
        key = "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"
        llm = OpenAI(api_key=key)
    elif "deepseekv3" in modelName:
        llm = OpenAI(api_key="421b4471-a00a-45ef-955e-050b3cc382f0",
                     base_url="https://ark.cn-beijing.volces.com/api/v3")
    else:
        if VLLM is False:
            ollama_base_url = "http://localhost:11434/"
            llm = OllamaLLM(
                base_url=ollama_base_url,
                model=modelName,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                    "stop": ["\n\n"]
                }
            )
        '''
        
        else:
            local_model_path = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
            llm = LLM(
                model=local_model_path,
                tensor_parallel_size=torch.cuda.device_count(),
                # max_num_batched_tokens=8192,
                max_model_len=8192,
                max_seq_len_to_capture=8192,
                # block_size=64,
                enable_chunked_prefill=False,
            )
        '''
    return llm
