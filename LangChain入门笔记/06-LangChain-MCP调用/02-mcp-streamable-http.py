import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

model = init_chat_model(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}}
)


async def run():
    # 1.同一份配置里混用两种 transport
    client = MultiServerMCPClient(
        {
            "time-mcp": {
                "transport": "stdio",
                "args": [
                    "-y",
                    "time-mcp"
                ],
                "command": "npx"
            },
            "12306-mcp": {
                "transport": "streamable_http",
                "url": "https://mcp.api-inference.modelscope.net/50719920b4f24d/mcp"
            }
        }
    )
    tools = await client.get_tools()

    # 2.创建 Agent，绑定模型和工具
    agent = create_agent(model=model, tools=tools)

    # 3.MCP 工具是异步的，这里必须用 ainvoke
    return await agent.ainvoke(
        {"messages": [HumanMessage("帮我查一下下周四从北京到杭州的高铁")]}
    )


response = asyncio.run(run())
for message in response["messages"]:
    message.pretty_print()
