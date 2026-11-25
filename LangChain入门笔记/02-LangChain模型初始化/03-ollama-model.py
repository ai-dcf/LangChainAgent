from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# 初始化 Ollama 上的本地模型
model = ChatOllama(
    model="qwen3.5:0.8b",
    temperature=1.0,
)

# 创建 Agent 并发送一条用户消息
agent = create_agent(model=model, tools=[])

response = agent.invoke({
    "messages": [{"role": "user", "content": "你是谁？"}]
})
print(response)
