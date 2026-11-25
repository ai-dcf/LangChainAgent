import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def create_bailian_model(model_name: str):
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )


for model_name in ("qwen3.8-flash", "qwen4-max"):
    model = create_bailian_model(model_name)
    print(model.model_name)
