# 7. Skill 调用

前面两节，我们学习了 Tool 和 MCP：模型可以调用工具完成计算、查询信息，也可以通过 MCP 连接其他系统提供的工具。

不过，有了工具，模型就一定知道如何完成一类任务吗？

例如，让 Agent 整理周报，它虽然可以阅读工作记录，但可能会把“正在联调”写成“已完成联调”，或者每次都使用不同的格式。我们可以在系统提示词中规定整理步骤、归类规则和输出格式。

如果以后还要写会议纪要、检查代码、分析数据，把所有规则都放进系统提示词，就会越来越长。每次只做一个任务，也要把其他任务的说明一起发给模型。

**Skill（技能）**就是把某类任务的处理方法、参考资料和可选脚本整理成一个可复用的目录，供 Agent 在需要时读取。它让 Agent 知道这类任务应该怎么做。[LangChain 的 Skills 文档](https://docs.langchain.com/oss/python/langchain/multi-agent/skills)介绍了通过工具按需加载专业提示词的实现方式。

## 7.1 Skill、Tool 和 MCP 的关系

| 概念 | 解决的问题 | 周报场景中的例子 |
| --- | --- | --- |
| Tool | Agent 能执行什么操作 | 读取参考文件、查询任务进度 |
| MCP | AI 应用如何连接外部服务提供的工具等资源 | 通过 MCP 获取项目管理系统中的任务 |
| Skill | 一类任务按照什么步骤、规则完成 | 先区分完成与进行中，再按模板整理周报 |

它们可以配合使用：Agent 读取“周报整理”技能，按照其中的流程调用 MCP 提供的任务查询工具，最后输出周报。

本节先用用户提供的虚构工作记录作为输入，不连接额外的 MCP 服务，便于观察 Skill 本身的加载过程。

> **注意**：Skill 不会自动变成一个可以执行的 Python 函数。所谓“调用 Skill”，通常是 Agent 通过工具读取说明，再按照说明工作。读取说明不会凭空注册工具，也不会自动执行目录中的脚本。
>
> Skill 的步骤属于模型遵循的提示词，不能替代程序中的权限检查或确定性的业务校验。

## 7.2 Skill 的目录结构

本节使用一个 `weekly-report` 技能。文章与示例的目录如下：

```text
06-LangChain-Skill调用/
├── LangChain-Skill调用.md
├── notebooks/
│   ├── 01-Skill-手动实现.ipynb
│   ├── 02-Skill-SkillsMiddleware.ipynb
│   └── 03-Skill-DeepAgents.ipynb
└── skills/
    └── weekly-report/
        ├── SKILL.md
        └── references/
            └── report-format.md
```

按照 [Agent Skills 格式规范](https://agentskills.io/specification)，技能目录中必须有 `SKILL.md`，也可以包含 `references/`（参考文档）、`scripts/`（脚本）、`assets/`（模板等资源）。本例只需要说明和参考文档。

### 7.2.1 SKILL.md

`SKILL.md` 包含两部分：前面的 YAML 元数据，和后面的 Markdown 正文。

```markdown
---
name: weekly-report
description: 将零散工作记录整理为中文周报。当用户要求写周报、汇总本周工作或整理下周计划时使用。
---

# 周报整理

## 执行步骤

1. 阅读用户提供的工作记录，区分已完成、进行中、问题风险和下周计划。
2. 生成周报前，先读取本技能目录内的 `references/report-format.md`，遵循其中的格式和归类规则。
3. 按模板整理周报，合并重复事项，保留输入中的数量、状态和明确依赖。
4. 只使用用户提供的事实，不编造完成情况、指标、负责人或截止日期。
5. 缺少的信息标记为“待补充”；进行中的事项不能写成已经完成。
6. 直接在回复中输出 Markdown 周报，无需写入文件，也无需调用其他智能体。

## 读取参考文件

- 手动实现：调用 `read_skill_file`，传入本技能名称和相对路径。
- nb-langchain-agentskills：调用 `skill__read_content`，传入本技能名称和相对路径。
- Deep Agents：调用 `read_file`，路径为当前 SKILL.md 所在目录下的 `references/report-format.md`。

工具名称由宿主提供。以上是本教程三种宿主的对应关系；Skill 文件本身不会注册或执行工具。
```

其中：

- `name`：技能名称。与技能目录同名，最多 64 个字符，使用小写字母、数字和连字符，不能以连字符开头、结尾或包含连续连字符。
- `description`：说明技能做什么、什么时候使用，最多 1024 个字符。模型主要靠这段描述判断是否应该加载技能。
- 正文：具体步骤、约束，以及需要读取的参考资料。

本例正文列出了三种宿主的工具名称，便于共用同一份文件；实际项目也可以用宿主通用的“读取相对路径文件”表述。工具由宿主代码提供。

### 7.2.2 参考资料

`references/report-format.md` 内容如下：

```markdown
# 周报格式与归类规则

输出必须包含以下三个二级标题：

## 本周完成

- 只列出明确已完成的事项，保留原始数量，不把计划或进行中工作列为已完成。
- 没有明确成果时写“待补充”。

## 问题风险

- 记录阻塞、依赖和未完成事项，并注明当前状态。
- 只有输入明确写了风险已解决，才能标记为已解决。
- 没有相关信息时写“待补充”，不要自行断言“无风险”。

## 下周计划

- 只列出输入中明确提出的后续安排。
- 可以整理表达，但不能补造负责人、日期、数量或额外任务。
- 没有相关信息时写“待补充”。

每一项使用简短的项目符号，不要添加绩效评价、寒暄或没有依据的总结。
```

为什么不全部写进 `SKILL.md`？这里为了演示第二次按需读取，把格式单独放在参考文件中。实际技能可以放入更长的规范、字段说明或案例，在需要时再读取。

## 7.3 渐进式加载

Skill 常用的加载方式称为**渐进式披露（Progressive Disclosure）**，也可以理解为“用到哪一层，读取哪一层”。

| 阶段 | 模型看到什么 | 本例中的内容 |
| --- | --- | --- |
| 发现 | 技能名称与描述，宿主也可能附带位置 | weekly-report 及其适用场景 |
| 激活 | 完整的技能说明 | SKILL.md 中的整理步骤 |
| 使用资料 | 当前任务需要的参考文件 | report-format.md 的格式规则 |

本例的调用过程可以这样理解：

```text
用户：把这些工作记录整理成周报
  ↓
模型根据技能描述选择 weekly-report
  ↓ 调用加载工具
技能说明作为 ToolMessage 加入上下文
  ↓ 模型按照说明调用读取工具
周报格式作为 ToolMessage 加入上下文
  ↓
模型根据记录和格式生成周报
```

这里“按需加载”说的是**什么时候把内容发送给模型**。Python 程序扫描技能时可能已经读取过文件来解析元数据，这不代表整个文件已经进入模型上下文。

读取后的内容会保留在当前消息历史中；本例没有实现自动卸载或上下文压缩，也不保证比一次性提示词消耗更少的总 Token。

下面分别用三种方式实现同一个过程。

## 7.4 运行准备

在项目根目录执行：

```powershell
uv sync
```

项目已声明本节需要的依赖。如果在其他项目复现，可以执行：

```powershell
uv add langchain langchain-deepseek python-dotenv pyyaml nb-langchain-agentskills deepagents
```

在项目根目录的 `.env` 中配置自己的密钥：

```dotenv
DEEPSEEK_API_KEY=你的密钥
```

Notebook 请选择本项目 `.venv` 对应的 Python 内核。下列示例使用本地文本读取，不需要前一节 MCP 的子进程和事件循环处理。

本节开发时安装的版本为 `nb-langchain-agentskills 0.9.2`、`deepagents 0.7.38`、`PyYAML 6.0.3`，项目实际解析版本以 `uv.lock` 为准。

三种方式的模型和路径初始化代码相同，每个 Notebook 都已经包含这一段，无需先运行另一个 Notebook：

```python
# 1.定位项目和本节目录，加载项目根目录的 .env
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# Notebook 中没有 __file__，从当前目录向上查找项目标记。
PROJECT_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents)
     if (path / "pyproject.toml").is_file()
     and (path / "LangChain入门笔记").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("请从项目根目录或本节 notebooks 目录启动 Notebook。")

CHAPTER_DIR = PROJECT_ROOT / "LangChain入门笔记" / "06-LangChain-Skill调用"
SKILLS_DIR = CHAPTER_DIR / "skills"
load_dotenv(PROJECT_ROOT / ".env")

# 2.沿用前文的 DeepSeek 模型；超时和重试次数用于限制演示等待时间。
model = init_chat_model(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
    timeout=60,
    max_retries=0,
)
```

从项目根目录、本节目录或 `notebooks` 目录运行，都可以向上找到项目。若运行环境无法访问该模型，请按前文的模型初始化方法更换为支持工具调用的模型。

## 7.5 方式一：手动实现

先用前面已经学过的 `create_agent` 和 `@tool` 实现，这样可以看清 Skill 的加载原理。

需要完成三件事：

1. 扫描技能文件，从 YAML 中提取名称与描述，生成技能目录。
2. 提供 `load_skill` 和 `read_skill_file`，让模型读取正文和参考资料。
3. 把技能目录放入系统提示词，再把读取工具绑定到 Agent。

接在初始化代码后运行：

```python
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
```

### 7.5.1 关键代码说明

- `yaml.safe_load`：解析 YAML 元数据，不把 YAML 当作任意 Python 对象执行。
- `skill_index`：保存名称、描述和本地路径；正文没有拼入初始提示词。
- `load_skill`：接收技能名称，返回 Markdown 正文。工具返回值会进入 Agent 的消息历史。
- `read_skill_file`：读取技能内的参考资料。通过 `resolve()` 和 `is_relative_to()` 检查最终路径，拒绝通过 `../` 或符号链接读取目录外文件。
- `create_agent(tools=...)`：注册的是两个读取工具，Skill 的步骤本身由模型理解和执行。

该实现只支持本例使用的简单 frontmatter 格式和一级技能目录，是为了讲清原理而写的教学版本，不是完整的 Agent Skills 规范实现。新增技能后需要重新运行扫描单元。

### 7.5.2 调用 Agent

三套示例使用相同的虚构记录：

```python
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
```

打印全部消息，是为了看到模型的 `tool_calls` 和对应的 `ToolMessage`。重点检查模型是否调用了：

```text
load_skill(skill_name="weekly-report")
read_skill_file(skill_name="weekly-report", file_path="references/report-format.md")
```

以上是**预期调用示意**，不是硬编码的执行顺序。是否调用以及具体参数由模型决定；若没有加载，先检查工具调用支持、技能描述以及系统提示词。

只查看最终周报时，可以使用：

```python
print(response["messages"][-1].content)
```

## 7.6 方式二：使用第三方 SkillsMiddleware

手动方式需要自己写扫描、加载和读取工具。如果希望继续使用 LangChain 的 `create_agent`，也可以使用第三方包 [nb-langchain-agentskills](https://pypi.org/project/nb-langchain-agentskills/)。

> 本节的 `SkillsMiddleware` 来自 `nb_langchain_agentskills`，不是 LangChain 核心包提供的类，也不是 Deep Agents 内部的同名中间件。

在一个新 Notebook 中运行 7.4 的初始化代码，然后运行：

```python
# 3.用第三方包发现技能，并向 Agent 注入目录和读取工具
from langchain.agents import create_agent
from nb_langchain_agentskills import DirectorySkillLoader, SkillsMiddleware

loader = DirectorySkillLoader(SKILLS_DIR)
skills_middleware = SkillsMiddleware(
    loader=loader,
    # 本例只需要读取说明和模板，不需要执行 shell 命令。
    exclude_tools={"skill__execute_script"},
)
print("可用技能：", [skill.name for skill in loader.list_skills()])
print("中间件提供的工具：", [tool.name for tool in skills_middleware.tools])

agent = create_agent(
    model=model,
    middleware=[skills_middleware],
    system_prompt="你是一个中文工作助手。任务匹配技能时先加载技能，按技能说明读取参考资料，仅依据用户事实回答。",
)
```

最后执行 7.5.2 的调用代码即可；对应的完整文件是 `notebooks/02-Skill-SkillsMiddleware.ipynb`。

### 7.6.1 中间件做了什么

我们没有手写技能目录和读取工具，因为中间件完成了这些工作：

- `DirectorySkillLoader`：扫描本地目录，发现并读取技能。
- `SkillsMiddleware`：把技能清单注入模型请求，注册相关工具。
- `middleware=[skills_middleware]`：将中间件接入 `create_agent` 的执行过程。

该包默认提供四个工具：

| 工具名称 | 用途 |
| --- | --- |
| `skill__list_skills` | 查看可用技能 |
| `skill__load_skill` | 加载指定技能的说明和目录信息 |
| `skill__read_content` | 读取技能中的文件 |
| `skill__execute_script` | 执行 shell 命令 |

本例通过 `exclude_tools` 排除了脚本执行工具，最终只暴露三个读取相关工具。不要把“限制文件目录”理解为 shell 沙箱；如果启用该包的执行工具，需要单独考虑命令执行权限。

### 7.6.2 观察调用过程

打印消息后，检查 `skill__load_skill` 是否加载了 `weekly-report`，以及 `skill__read_content` 是否读取了 `references/report-format.md`。

加载工具返回的内容比手动实现丰富，但核心过程相同：先知道有哪些技能，再读取相关说明，最后读取参考资料。

## 7.7 方式三：使用 Deep Agents 内置支持

如果还需要文件系统工具、任务规划、上下文管理等能力，可以使用 LangChain 生态中的 Deep Agents。它通过 `create_deep_agent` 提供更完整的 Agent 执行环境，内置 Skill 支持，使用方式见 [Deep Agents 官方文档](https://docs.langchain.com/oss/python/deepagents/skills)。

在一个新 Notebook 中运行 7.4 的初始化代码，然后运行：

```python
# 3.使用 Deep Agents 内置的技能发现和文件读取能力
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission

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
```

最后仍然使用 7.5.2 的相同调用代码。完整文件是 `notebooks/03-Skill-DeepAgents.ipynb`。

### 7.7.1 参数说明

| 参数 | 本例中的作用 |
| --- | --- |
| `model` | 使用与前两种方式相同的模型对象 |
| `backend` | 通过 FilesystemBackend 读取本地技能文件 |
| `root_dir` | 让虚拟路径的根目录对应本节目录 |
| `virtual_mode=True` | 文件路径限制在 Backend 根目录内 |
| `skills=["/skills/"]` | 指向包含技能子目录的父目录 |
| `permissions` | 禁止文件工具执行写操作，周报直接在回复中输出 |

Windows 用户也使用 `/skills/` 这样的正斜杠虚拟路径。这里的 `/` 对应 `CHAPTER_DIR`，不是磁盘根目录。

不要把 `skills` 写成单个技能的 `/skills/weekly-report/`；该参数接收技能源目录，本例需要扫描它下面的技能子目录。

### 7.7.2 观察调用过程

Deep Agents 内部会使用自己的 `SkillsMiddleware` 处理技能发现。这里不用自行导入或同时添加上一节的第三方中间件。

打印消息时，重点查看 `read_file` 是否先读取：

```text
/skills/weekly-report/SKILL.md
```

然后读取：

```text
/skills/weekly-report/references/report-format.md
```

Deep Agents 还提供其他工具，但这次任务只需要读取说明、读取模板和生成回复，不需要创建文件或委派任务。不要把本例的 Backend 路径限制当作任意代码执行沙箱。

## 7.8 结果检查与常见问题

三种方式的最终周报都应该包含“本周完成”“问题风险”“下周计划”。下面只是**格式示意**，实际措辞以运行结果为准：

```markdown
## 本周完成
- 完成登录接口联调，修复 2 个参数校验问题。
- 完成 MCP 调用笔记初稿。

## 问题风险
- Skill 调用示例仍在编写，尚未完成。
- 测试环境偶发超时，正在等待运维排查，尚未解决。

## 下周计划
- 完成 Skill 示例。
- 补充登录接口的异常场景测试。
```

除了看输出格式，还要确认“进行中”没有被改成“已完成”、“等待排查”没有被改成“已解决”。仅有三个标题并不能证明模型遵守了事实约束。

| 现象 | 检查方向 |
| --- | --- |
| 找不到技能 | 检查目录、SKILL.md 文件名和元数据；Deep Agents 检查技能源虚拟路径 |
| 没有调用加载工具 | 检查模型是否支持工具调用，description 是否明确，系统提示词是否要求匹配时先加载 |
| 没有读取参考文件 | 检查正文是否要求读取、文件相对路径是否正确，并查看完整消息 |
| API 调用失败 | 检查密钥、网络、账户余额、模型名称，错误不代表技能文件一定有问题 |
| Notebook 导入失败 | 在项目根目录 uv sync，并选择项目内核，必要时重启内核 |

### 7.8.1 本次验证记录

2026 年 11 月 14 日，在本项目 Windows 环境中，使用 `deepseek-v4-flash` 对三个 Notebook 分别进行了真实调用，结果如下：

| 方式 | 实际观察到的读取过程 | 最终结果 |
| --- | --- | --- |
| 手动实现 | `load_skill` → `read_skill_file` | 三个栏目完整，保留了未完成、未解决状态 |
| 第三方中间件 | `skill__load_skill` → `skill__read_content` | 三个栏目完整，保留了未完成、未解决状态 |
| Deep Agents | `read_file` 读取 SKILL.md → `read_file` 读取格式参考文件 | 三个栏目完整，保留了未完成、未解决状态 |

三个 Notebook 已保存本次真实运行的消息输出。上面的调用结果说明本次运行通过，不保证模型每次都使用完全相同的措辞。

另外，Notebook 格式与代码语法、根目录和 Notebook 目录下的初始化、未知技能与越界路径处理、依赖兼容性均通过检查。通过离线模型捕获首轮请求，确认三种方式都只提供了技能清单，没有提前将技能正文或格式参考文件放入模型上下文。

## 7.9 总结

| 方式 | 需要自己实现的部分 | 适合的场景 |
| --- | --- | --- |
| 手动实现 | 元数据解析、目录、读取工具 | 学习原理，少量且可控的技能 |
| 第三方 SkillsMiddleware | 配置 Loader 和中间件 | 保留 create_agent，复用技能管理能力 |
| Deep Agents | 配置 Backend 和 skills | 同时需要技能及更完整的 Agent 配套能力 |

三种方式共用同一份技能文件，说明可以复用的是任务方法与参考资料，而扫描、读取、工具名称等能力仍取决于宿主。

记住三个要点：Tool 提供操作，Skill 提供方法；Skill 通常通过工具读取后生效；先传名称和描述、再按需传正文与参考资料，可以避免在初始提示中塞入所有任务说明。

参考资料：

- [LangChain Skills 模式](https://docs.langchain.com/oss/python/langchain/multi-agent/skills)
- [Agent Skills 格式规范](https://agentskills.io/specification)
- [nb-langchain-agentskills 项目说明](https://pypi.org/project/nb-langchain-agentskills/)
- [Deep Agents Skills](https://docs.langchain.com/oss/python/deepagents/skills)
