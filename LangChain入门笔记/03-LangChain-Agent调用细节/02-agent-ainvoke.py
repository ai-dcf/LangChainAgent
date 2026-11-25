import asyncio

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


async def main():
    response = await agent.ainvoke({
        "messages": [HumanMessage("你是谁？")]
    })
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
