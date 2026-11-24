"""直接使用 ChatOllama 初始化本地模型。

运行前启动 Ollama，并执行 ollama pull qwen3.5:0.8b。
"""

from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3.5:0.8b", temperature=1.0)
response = model.invoke("用一句话介绍你自己。")
print(response.text)
