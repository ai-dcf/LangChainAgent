from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

# 2.调用init_chat_model初始化模型
model = init_chat_model(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}}
)

# 3.初始化Agent
agent = create_agent(model = model, tools=[])

# 4.调用Agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "你是谁？"}]
})

# 5.打印大模型消息
messages = response["messages"]
for message in messages:
    message.pretty_print()