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

for chunk in agent.stream(
    {"messages": [HumanMessage("你是谁？")]},
    stream_mode="messages",
    version="v2",
):
    token, _ = chunk["data"]
    print(token.text, end="", flush=True)
