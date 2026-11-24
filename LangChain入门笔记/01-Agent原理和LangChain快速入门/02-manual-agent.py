import datetime
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


def current_time() -> str:
    """获取当前时间。"""
    return str(datetime.datetime.now())


tools_by_name = {"current_time": current_time}
tools = [
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "获取当前时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            name = call.function.name
            tool = tools_by_name.get(name)
            if tool is None:
                raise ValueError(f"未注册的工具：{name}")

            arguments = json.loads(call.function.arguments)
            print(f"工具请求：{name}，参数：{call.function.arguments}")
            result = tool(**arguments)
            print(f"工具结果：{result}")

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    raise RuntimeError("达到 5 次模型调用，仍未生成最终回答。")


print(f"最终回答：{run_agent('今天是几月几号')}")
