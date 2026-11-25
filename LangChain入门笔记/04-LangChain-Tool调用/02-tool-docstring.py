from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool


@tool
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """查询指定城市的当前天气，也可以附带未来几天的预报。

    Args:
        location: 城市名，例如"杭州"
        units: 温度单位，celsius 或 fahrenheit
        include_forecast: 是否需要未来五天的预报
    """
    temp = 22 if units == "celsius" else 72
    result = f"{location} 当前温度：{temp} {units}"
    if include_forecast:
        result += "\n未来五天：晴"
    return result


print("工具名：", get_weather.name)
print("用途：", get_weather.description)
print("参数：", get_weather.args)
print("直接调用：", get_weather.invoke({"location": "杭州", "include_forecast": True}))

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)
agent = create_agent(model=model, tools=[get_weather])

for step in agent.stream(
    {"messages": [HumanMessage("杭州现在冷不冷？顺便说说未来几天的天气。")]},
    stream_mode="updates",
):
    for messages in step.values():
        for message in messages:
            if isinstance(message, ToolMessage):
                print("工具返回：", message.content[:200])
