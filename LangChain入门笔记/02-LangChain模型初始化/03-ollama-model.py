from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=1.0,
)

response = model.invoke("你是谁？")
print(response.text)
