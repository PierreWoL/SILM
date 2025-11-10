from openai import OpenAI
from langchain_ollama import OllamaLLM


def chooseLLM(model_name):
    if "deepseekV3" in model_name:
        client = OpenAI(
            api_key="",
            base_url="",
        )
    elif "gpt" in model_name:
        key = ("")
        client = OpenAI(api_key=key)
    else:
        ollama_base_url = "http://localhost:11434/"
        client = OllamaLLM(
            base_url=ollama_base_url,
            model=model_name,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 8192,
                "stop": ["\n\n"]
            }
        )
    return client