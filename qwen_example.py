from openai import OpenAI
import os

client = OpenAI(
    # If the environment variable is not set, replace it with your Model Studio API key: api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://[workspace-id].ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "Who are you"}]
completion = client.chat.completions.create(
    model="qwen3.7-plus",  # You can replace this with another deep thinking models
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
is_answering = False  # Indicates whether the response phase has started
print("\n" + "=" * 20 + "Thinking process" + "=" * 20)
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Full response" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)