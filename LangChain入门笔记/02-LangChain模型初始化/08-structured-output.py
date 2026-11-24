"""用 Pydantic 定义结构化输出。"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class CapitalInfo(BaseModel):
    name: str = Field(description="城市名称")
    location: str = Field(description="城市在月球上的位置")
    vibe: str = Field(description="城市的风格和生活氛围")
    economy: str = Field(description="城市的主要产业和经济来源")


load_dotenv()

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

structured_model = model.with_structured_output(CapitalInfo, method="function_calling")
capital = structured_model.invoke("为科幻小说设计一座虚构的月球首都，用中文填写每个字段。")

if capital is None:
    raise SystemExit("模型未返回月球首都的结构化数据。")
print(capital.model_dump_json(indent=2))
