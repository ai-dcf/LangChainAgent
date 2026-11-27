# 1.定位项目和本节目录，加载项目根目录的 .env
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 脚本使用 __file__ 定位，不依赖当前工作目录。
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (path for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (path / "pyproject.toml").is_file()
     and (path / "LangChain入门笔记").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("请在本项目的 LangChain入门笔记 目录中运行脚本。")

CHAPTER_DIR = SCRIPT_PATH.parent
SKILLS_DIR = CHAPTER_DIR / "skills"
load_dotenv(PROJECT_ROOT / ".env")

# 2.沿用前文的 DeepSeek 模型；超时和重试次数用于限制演示等待时间。
model = init_chat_model(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max_retries=0,
)

# 3.使用 Deep Agents 内置的技能发现和文件读取能力
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission

# SKILLS_DIR 在上面统一初始化；Deep Agents 通过 Backend 的虚拟路径访问技能。
# 虚拟路径 / 对应本节目录，不使用 Windows 绝对路径作为 skills 参数。
backend = FilesystemBackend(root_dir=CHAPTER_DIR, virtual_mode=True)
agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=["/skills/"],
    # 演示只读技能，最终周报直接回复；通过文件工具禁止写入本节文件。
    permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
    system_prompt=(
        "你是一个中文工作助手。任务匹配技能时先读取对应 SKILL.md，再读取说明要求的参考资料。"
        "仅依据用户事实回答，直接在回复中输出周报。请自行完成，不调用 task 委派任务。"
    ),
)

# 6.使用相同的虚构记录调用 Agent，打印完整消息观察加载过程
work_notes = """请把下面的工作记录整理为本周周报。
- 本周完成登录接口联调，修复了 2 个参数校验问题。
- 完成 MCP 调用笔记的初稿。
- Skill 调用示例仍在编写，尚未完成。
- 测试环境偶发超时，正在等待运维排查，问题尚未解决。
- 下周计划完成 Skill 示例，并补充登录接口的异常场景测试。
"""
response = agent.invoke(
    {"messages": [HumanMessage(work_notes)]},
    config={"recursion_limit": 30},
)
for message in response["messages"]:
    message.pretty_print()
print(response["messages"][-1].content)
