from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilySearch(max_results=5, topic="general")

result = tavily.invoke({"query": "杭州今天天气如何？"})
if isinstance(result, dict):
    for item in result.get("results", []):
        print(item["title"])
        print(item["url"])
        print(item.get("content", "")[:200])
else:
    print(result)

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)
agent = create_agent(
    model=model,
    tools=[tavily],
    system_prompt="涉及实时信息时先使用网页搜索，并在回答中注明来源。",
)

for step in agent.stream(
    {"messages": [HumanMessage("请搜索北京未来五天的天气预报。")]},
    stream_mode="updates",
):
    for messages in step.values():
        for message in messages:
            if isinstance(message, ToolMessage):
                print("工具返回：", message.content[:300])
