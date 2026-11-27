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

# 3.发现本地技能：仅将名称、描述和路径留在索引中
import re
import yaml
from langchain.agents import create_agent
from langchain.tools import tool


def parse_skill(path: Path) -> tuple[dict, str]:
    """拆分 YAML 元数据和 Markdown 正文。"""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"{path.name} 必须以 YAML frontmatter 开头。")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError("Skill 元数据必须是 YAML 映射。")
    name = metadata.get("name")
    description = metadata.get("description")
    if (not isinstance(name, str) or len(name) > 64
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
        or name != path.parent.name):
        raise ValueError("name 必须与目录同名，使用小写字母、数字和单个内部连字符，最长 64 字符。")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1024:
        raise ValueError("description 必须是 1～1024 字符的非空字符串。")
    body = parts[2].strip()
    if not body:
        raise ValueError("Skill 正文不能为空。")
    return metadata, body


skill_index = {}
# 入门示例只扫描 skills 的直接子目录，不实现递归发现或热刷新。
for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
    root = skill_file.parent.resolve()
    if not root.is_relative_to(SKILLS_DIR.resolve()):
        raise ValueError("技能目录不能指向 skills 目录以外。")
    metadata, _ = parse_skill(skill_file)
    name = metadata["name"]
    if name in skill_index:
        raise ValueError(f"技能名称重复：{name}")
    skill_index[name] = {
        "description": metadata["description"],
        "root": root,
    }

if not skill_index:
    raise RuntimeError("没有发现技能，请检查 skills 目录。")


# 4.定义工具：模型通过工具把正文、参考资料加入消息上下文
@tool
def load_skill(skill_name: str) -> str:
    """按技能名称加载完整 SKILL.md 正文；使用技能前必须先调用。"""
    entry = skill_index.get(skill_name)
    if entry is None:
        return f"错误：未知技能 {skill_name}。可用技能：{', '.join(skill_index)}"
    try:
        skill_file = (entry["root"] / "SKILL.md").resolve()
        if not skill_file.is_relative_to(entry["root"]):
            return "错误：技能文件不能超出技能目录。"
        _, body = parse_skill(skill_file)
        return body
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return f"错误：无法加载技能（{type(exc).__name__}）。"


@tool
def read_skill_file(skill_name: str, file_path: str) -> str:
    """读取指定技能内的 UTF-8 参考文件，file_path 必须是相对技能目录的路径。"""
    entry = skill_index.get(skill_name)
    if entry is None:
        return f"错误：未知技能 {skill_name}。"
    relative = Path(file_path)
    if relative.is_absolute() or relative.drive or relative.root:
        return "错误：请使用技能目录内的相对路径。"
    try:
        target = (entry["root"] / relative).resolve()
        if not target.is_relative_to(entry["root"]):
            return "错误：文件路径不能超出技能目录。"
        if not target.is_file():
            return "错误：参考文件不存在。"
        return target.read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeError) as exc:
        return f"错误：无法读取参考文件（{type(exc).__name__}）。"


# 5.初始提示词只有技能目录，没有正文或模板
catalog = "\n".join(
    f"- {name}: {entry['description']}" for name, entry in skill_index.items()
)
system_prompt = (
    "你是一个中文工作助手。以下是可用技能的名称和描述：\n"
    + catalog
    + "\n任务匹配技能时，先用 load_skill 加载说明，再按说明使用工具读取所需资料。"
    "仅依据用户提供的事实回答。"
)
agent = create_agent(
    model=model,
    tools=[load_skill, read_skill_file],
    system_prompt=system_prompt,
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
