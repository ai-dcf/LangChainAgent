import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
bailian_model = init_chat_model(
    model="qwen3.8-flash",
    model_provider="openai",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
print(type(bailian_model).__name__)
