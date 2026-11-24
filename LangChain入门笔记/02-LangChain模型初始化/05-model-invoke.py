"""用 invoke 一次拿到完整的模型回复。"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

response = model.invoke("用两句话解释 LangChain 的聊天模型。")
print(response.text)
