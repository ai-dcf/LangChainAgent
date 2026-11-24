import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

# 1.加载环境变量
load_dotenv()

base_url = os.getenv("DASHSCOPE_BASE_URL")
api_key = os.getenv("DASHSCOPE_API_KEY")

model_names = ["qwen4-max", "qwen3.8-flash"]

# 2.定义创建模型的函数
def create_model(model_name:str):
    return init_chat_model(
        model=model_name, # 模型名称，这里可以自定义，我们用的是阿里的qwen-max
        model_provider="openai", # 如果是langchain不支持的模型，用openai的规范
        base_url=base_url,
        api_key=api_key
    )

for model_name in model_names:
    # 3.初始化Agent，传入初始化好的model、tools，tools可以为空
    agent = create_agent(model = create_model(model_name), tools=[])

    # 4.调用Agent
    response = agent.invoke({
        "messages": [{"role": "user", "content": "你是谁？"}]
    })

    # 5.打印大模型消息
    messages = response["messages"]
    for message in messages:
        message.pretty_print()