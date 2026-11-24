import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

load_dotenv()

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)


@tool
def current_time() -> str:
    """获取当前时间"""
    return str(datetime.datetime.now())


agent = create_agent(model=model, tools=[current_time])
response = agent.invoke({"messages": [HumanMessage("今天是几月几号")]})

for message in response["messages"]:
    message.pretty_print()
