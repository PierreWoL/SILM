from openai import OpenAI


def get_chatgpt_response(gpt_model, messages, is4=False):
    if is4 is True:
        model = "gpt-3.5-turbo-1106"
    else:
        model = "gpt-4-1106-preview"
    response = gpt_model.chat.completions.create(
        model=model,
        messages=messages)
    new_assistant_reply = response.choices[0].message.content
    return new_assistant_reply


def get_llama_answer(model, messages):
    response = model.invoke(messages)
    return response


def get_model_answer(model, messages, is4=False):
    response = ""
    if isinstance(model, OpenAI):
        response = get_chatgpt_response(model, messages, is4=is4)
    else:
        response = get_llama_answer(model, messages)
    return response
