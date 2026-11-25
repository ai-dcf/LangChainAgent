import os
import subprocess
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.tools import ToolException
from pprint import pprint

load_dotenv()

WORKDIR = Path(tempfile.mkdtemp(prefix="langchain-tool-demo-"))
PDF_PATH = Path("LangChain入门笔记/02-LangChain模型初始化/files/太平洋小蜜蜂6号综合意外险条款.pdf")


@tool
def deepseek_web_search(query: str, count: int = 5) -> dict:
    """用 DeepSeek 搜索网页，适合查询实时信息、核实时效性事实。

    Args:
        query: 搜索关键词，支持 site: 等搜索语法
        count: 返回的结果条数，最多 50 条
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ToolException("请先在 .env 中配置 DEEPSEEK_API_KEY。")

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/tools/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "count": count, "need_content": False},
            timeout=60,
        )
        response.raise_for_status()
        results = response.json()["results"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise ToolException(f"搜索失败：{exc}") from exc

    return {
        "results": [
            {
                "title": item["title"],
                "url": item["url"],
                "summary": item["summary"],
                "date": item["date"],
            }
            for item in results
        ]
    }


@tool
def deepseek_ocr(file_path: str, purpose: str = "text_recognition") -> dict:
    """识别 PDF、Word、PPT 或图片中的文字，按页返回结果。

    Args:
        file_path: 要识别的文件路径，相对项目根目录
        purpose: text_recognition 或 rich_layout_recognition
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ToolException("请先在 .env 中配置 DEEPSEEK_API_KEY。")

    path = Path(file_path)
    if not path.is_file():
        raise ToolException(f"文件不存在：{file_path}")

    try:
        with path.open("rb") as file_obj:
            response = requests.post(
                "https://api.deepseek.com/v1/tools/ocr",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (path.name, file_obj)},
                data={"purpose": purpose},
                timeout=180,
            )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ToolException(f"OCR 请求失败：{exc}") from exc

    if "error" in result:
        raise ToolException(f"OCR 识别失败：{result['error']}")

    return {
        "page_count": result["page_count"],
        "pages": [
            {
                "index": page["index"],
                "text": page["text"][:500],
                "finish_reason": page["finish_reason"],
            }
            for page in result["pages"]
        ],
    }


@tool
def read_file(path: str) -> str:
    """读取临时工作目录中的文本文件。

    Args:
        path: 文件名，例如 note.txt
    """
    return (WORKDIR / path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """把文本写入临时工作目录，文件不存在时创建。

    Args:
        path: 文件名，例如 note.txt
        content: 要写入的文本
    """
    target = WORKDIR / path
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}"


@tool
def edit_file(path: str, old: str, new: str) -> str:
    """把文件里的一段文本替换成另一段文本。

    Args:
        path: 文件名
        old: 原文本
        new: 新文本
    """
    target = WORKDIR / path
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace(old, new), encoding="utf-8")
    return f"已修改 {path}"


@tool
def delete_file(path: str) -> str:
    """删除临时工作目录中的文件。

    Args:
        path: 文件名
    """
    (WORKDIR / path).unlink()
    return f"已删除 {path}"


@tool
def run_command(command: list[str]) -> dict:
    """在临时工作目录里执行一条命令，返回退出码和输出。

    Args:
        command: 命令及参数组成的列表，例如 ["echo", "hello"]
    """
    completed = subprocess.run(
        command,
        cwd=WORKDIR,
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[:4000],
        "stderr": completed.stderr[:4000],
    }


tools = [
    deepseek_web_search,
    deepseek_ocr,
    read_file,
    write_file,
    edit_file,
    delete_file,
    run_command,
]
for tool_obj in tools:
    tool_obj.handle_tool_error = True


print("搜索结果：")
pprint(deepseek_web_search.invoke({"query": "DeepSeek API 文档 site:api-docs.deepseek.com", "count": 3}))

print("OCR 结果：")
pprint(deepseek_ocr.invoke({"file_path": str(PDF_PATH)}))

print("文件工具：")
print(write_file.invoke({"path": "note.txt", "content": "你好，LangChain"}))
print(read_file.invoke({"path": "note.txt"}))
print(edit_file.invoke({"path": "note.txt", "old": "LangChain", "new": "Tool"}))
print(read_file.invoke({"path": "note.txt"}))
print(delete_file.invoke({"path": "note.txt"}))

print("命令执行：")
pprint(run_command.invoke({"command": ["echo", "hello"]}))

model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="需要使用工具时直接调用，完成后用中文简要说明结果。",
)
response = agent.invoke({
    "messages": [HumanMessage("在临时目录里写一个 hello.txt，内容写‘你好’，然后读出来。")]
})
print("Agent 回复：", response["messages"][-1].content)
