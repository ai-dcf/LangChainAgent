from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)
tools = []
agent = create_agent(model=model, tools=tools)

stream = agent.stream_events(
    {"messages": [HumanMessage("你是谁？")]},
    version="v3",
)
for message in stream.messages:
    for text in message.text:
        print(text, end="", flush=True)
