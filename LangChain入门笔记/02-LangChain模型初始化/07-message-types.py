"""用 SystemMessage 和 HumanMessage 构造输入，查看返回的 AIMessage。"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

messages = [
    SystemMessage(content="你是一位天文馆讲解员，用小学生能听懂的语言回答。"),
    HumanMessage(content="为什么月亮每天看起来形状不一样？"),
]
response = model.invoke(messages)
response.pretty_print()
