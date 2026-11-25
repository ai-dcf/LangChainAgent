from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
deepseek_model = init_chat_model(model="deepseek-flash")
print(type(deepseek_model).__name__)
