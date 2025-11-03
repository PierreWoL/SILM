from openai import OpenAI
from langchain_ollama import OllamaLLM


def chooseLLM(model_name):
    if "deepseekV3" in model_name:
        client = OpenAI(
            api_key="421b4471-a00a-45ef-955e-050b3cc382f0",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
    elif "gpt" in model_name:
        key = ("sk-proj-BlbaveLOKrB8iZgeMkZGyaOxj9fVg3arCRbw3C43xJwiv71t"
               "-hL68CqzlCHrQ4p0Bf729dyRb6T3BlbkFJN73IYqVygitljyeRUYOOWok"
               "-M0jPe4y2fN942x9_FYiwjcCB7lSaoOHDOQjj4EDo6yEfqMgFwA")
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