import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

pdf_path = Path(__file__).parent / "files" / "太平洋小蜜蜂6号综合意外险条款.pdf"

load_dotenv()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 上传 PDF，使用文字解析并保留一小时
with pdf_path.open("rb") as pdf:
    uploaded = client.files.create(
        purpose="text_recognition",
        file=(pdf_path.name, pdf, "application/pdf"),
        expires_after={"anchor": "created_at", "seconds": 3600},
    )

file_id = uploaded.id
print("file_id =", file_id)

# 在用户消息中引用文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "file", "file_id": file_id},
        {"type": "text", "text": "请根据文档首页列出条款正式名称、注册号和第一条标题。"},
    ],
}
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[message],
    extra_body={"thinking": {"type": "disabled"}},
)
print(response.choices[0].message.content)
