# 4. Store（长期记忆）

store是LangChain提供的长期记忆机制，用于在不同会话间共享数据。例如：模型以外的数据、用户偏好等。



例如，同一位用户可能先在 `thread-a` 中告诉 Agent“我喜欢代码示例优先”，几天后又在 `thread-b` 中开始新的课程咨询。

如果这条偏好只放在 State 中，新 thread 无法自动读取；把它写入 Store 后，两个 thread 都可以访问。



LangChain提供了多种Store的实现方式，例如：

* InMemoryStore

* PostgresStore

* RedisStore

* ...

课程中我们以InMemoryStore为例来学习。



## 4.1 Store的数据结构

Store的数据格式是JSON文档，JSON文档采用分级管理：

* Namespace（命名空间）：可以理解为一个文件夹

  * Key（键）：可以理解为文件名，必须唯一

  * Value（值）：要存储的JSON文档



所以，定位Store中的一条数据，需要两部分信息：`namespace + key`，其中`namespace`是一个元组，而`key`则是普通字符串。



例如，在多租户的在线教育Agent，我们要记录每个用户的学习偏好，namespace和key分别可以采用下面的组织方式。



| **namespace** | **key**                 | **value**                           |
| ------------- | ----------------------- | ----------------------------------- |
|               | "learning\_preferences" | {"内容顺序":"先写代码，再解释原理"} |
|               | "coding\_preferences"   | {"代码风格":"简洁但要有注释"}       |
|               | "learning\_preferences" | {"...":"..."}                       |
|               | "coding\_preferences"   | {"...":"..."}                       |



由于Store的结构是固定的`namespace + key -> json文档`的格式，所以无需我们定义结构，我们只要确定好自己要用的namespace和key，就可以直接使用。

```python
namespace = (
    "heima_platform",  # 黑马程序员平台
    "school_a",        # 租户id，例如某个校区
    "u002"             # 用户id，也就是学员
)

key = "learning_preferences" # key是普通字符串，这里表示学习偏好
```





## 4.2 初始化Store

首先，我们要初始化Store：

```python
from langgraph.store.memory import InMemoryStore

# ==================== 1. 创建Store ====================
store = InMemoryStore()
```



store中有两个方法，分别用来读取和写入Store：

* `store.get(namespace, key)`：读取Store

* `store.put(namespace, key, value)`：存入Store，这里的value必须是`dict[str,Any]`结构，而且dict中的值必须是可以转JSON的，LangChain会将其转JSON存储



## 4.3 在tool中访问store

与state类似，LangChain中访问store通常也可以有两种场景：

* Tool

* 中间件

我们依然先学习tool中访问store，与state类似，在tool 中访问store也是通过runtime



在tool中访问store要这样做：

```python
from langchain.tools import ToolRuntime, tool


# 定义方法，根据Context中的tenant_id和user_id拼接得到namespace
def preference_namespace(ctx: UserContext) -> tuple[str, ...]:
    return (
        "heima_platform",
        ctx.tenant_id,
        ctx.user_id
    )
# 定义固定的key
key = "learning_preferences"

@tool
def save_learning_preference(
    preference_name: str,
    preference_value: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """保存当前用户的长期学习偏好。"""

    # 通过上下文中的tenant_id和user_id来构建namespace
    namespace = preference_namespace(runtime.context)

    # 通过namespace和key来读取用户偏好, 返回结果item.value是dict格式
    item = runtime.store.get(namespace, key)
    preferences = dict(item.value) if item else {}

    # 记录本次要保存的用户偏好
    preferences[preference_name] = preference_value

    # 保存到Store中
    runtime.store.put(namespace, key, preferences)

    return f"已保存偏好：{preference_name}={preference_value}"


@tool
def get_learning_preferences(
    runtime: ToolRuntime[UserContext],
) -> str:
    """读取当前用户的长期学习偏好。"""

    namespace = preference_namespace(runtime.context)
    item = runtime.store.get(namespace, key)

    if item is None:
        return "当前用户还没有保存学习偏好"

    return f"当前学习偏好：{item.value}"
```





## 4.4 给agent添加store

创建agent时需要指定store：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


checkpointer = InMemorySaver()
store = InMemoryStore()

agent = create_agent(
    model="deepseek-flash",
    tools=[
        get_current_user_profile,
        save_learning_preference,
        get_learning_preferences,
    ],
    context_schema=UserContext,
    checkpointer=checkpointer,
    store=store,
    system_prompt=(
        "你是课程平台助手。"
        "涉及用户资料时调用资料工具，"
        "涉及长期学习偏好或给用户讲解代码时调用偏好工具。"
    ),
)
```



测试：

```python
context = UserContext(
    user_id="u001",
    tenant_id="school_a",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "记住：我喜欢先看代码，再看原理",
            }
        ]
    },
    config={"configurable": {"thread_id": "thread-a"}},
    context=context,
)
for message in result["messages"]:
    message.pretty_print()
```



运行结果：

```plain&#x20;text
================================ Human Message =================================

记住：我喜欢先看代码，再看原理
================================== Ai Message ==================================
Tool Calls:
  get_learning_preferences (call_00_8Rapj2rtpvKD6W9zdMgB6518)
 Call ID: call_00_8Rapj2rtpvKD6W9zdMgB6518
  Args:
  save_learning_preference (call_01_vXZjnAHR1GMP0KKGL24l0258)
 Call ID: call_01_vXZjnAHR1GMP0KKGL24l0258
  Args:
    preference_name: 学习内容顺序偏好
    preference_value: 先看代码示例，再看原理讲解
================================= Tool Message =================================
Name: get_learning_preferences

当前用户还没有保存学习偏好
================================= Tool Message =================================
Name: save_learning_preference

已保存偏好：学习内容顺序偏好=先看代码示例，再看原理讲解
================================== Ai Message ==================================

已记住 ✅

**偏好：学习内容顺序偏好 = 先看代码示例，再看原理讲解**

以后我给你讲任何知识点（新概念、算法、框架用法等），都会按这个顺序来：

1. 先上可运行的代码示例，让你直接看到「长什么样、怎么跑」
2. 再拆解背后的原理、设计动机和边界情况

如果某次你想临时调换顺序（比如先要原理概览），随时说一声就行，不会影响这条长期偏好。

需要我现在就按这个方式讲点什么吗？
```



```sql
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "langchain如何实现结构化输出？",
            }
        ]
    },
    config={"configurable": {"thread_id": "thread-b"}},
    context=context,
)

for message in result["messages"]:
    message.pretty_print()
```

输出结果：

````markdown
================================ Human Message =================================

langchain如何实现结构化输出？
================================== Ai Message ==================================
Tool Calls:
  get_current_user_profile (call_00_IWdwonCMHAK9hNJT9KcQ4899)
 Call ID: call_00_IWdwonCMHAK9hNJT9KcQ4899
  Args:
  get_learning_preferences (call_01_r13pQ0XkS6pIVqkO1jRN1052)
 Call ID: call_01_r13pQ0XkS6pIVqkO1jRN1052
  Args:
================================= Tool Message =================================
Name: get_current_user_profile

姓名：虎哥，课程：LangChain 开发指南，
================================= Tool Message =================================
Name: get_learning_preferences

当前学习偏好：{'内容顺序偏好': '先看代码示例，再看原理讲解'}
================================== Ai Message ==================================

虎哥，按你「先代码后原理」的习惯，先上可直接跑的示例，再讲底层机制。

## 一、代码示例

### 1. 最常用：`with_structured_output()`

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    """从文本中抽取的人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    hobbies: list[str] = Field(default_factory=list, description="爱好列表")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_llm = llm.with_structured_output(Person)   # 关键一步
result = structured_llm.invoke("张三今年 28 岁，喜欢爬山和摄影。")

print(result)              # Person(name='张三', age=28, hobbies=['爬山', '摄影'])
print(result.age, type(result.age))   # 28 <class 'int'>，已经是 Python 对象了
```

想同时拿到原始消息和解析错误（调试、埋点用）：

```python
structured_llm = llm.with_structured_output(Person, include_raw=True)
out = structured_llm.invoke("张三今年 28 岁")
out["raw"]             # AIMessage
out["parsed"]          # Person 对象（失败时为 None）
out["parsing_error"]   # None 或 ValidationError
```

换底层模式（不同模型走不同能力）：

```python
llm.with_structured_output(Person, method="function_calling")   # 默认，走工具调用
llm.with_structured_output(Person, method="json_mode")          # 走 response_format=json_object
llm.with_structured_output(Person, method="json_schema", strict=True)  # OpenAI 严格模式
```

不想要 Pydantic，也可以用 `TypedDict` 或 JSON Schema：

```python
from typing import TypedDict

class Movie(TypedDict):
    title: str
    year: int

structured_llm = llm.with_structured_output(Movie)
```

### 2. 放进 LCEL 链里

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息抽取助手。"),
    ("human", "{text}")
])

chain = prompt | structured_llm          # 输出直接是 Person 对象
chain.invoke({"text": "李四 35 岁，爱好下棋、跑步"})
```

### 3. 传统解析器方案（`PydanticOutputParser`）

老项目/不支持工具调用的模型走这条路：

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "按指定 JSON 格式回答。\n{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
chain.invoke({"text": "张三 28 岁，爱好爬山、摄影"})
```

要流式输出部分 JSON 字段，用 `JsonOutputParser`：

```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=Person)
for chunk in (prompt | llm | parser).stream({"text": "..."}):
    print(chunk)      # 逐步吐出不完整的 dict
```

### 4. Agent 的结构化输出

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[], response_format=Person)
agent.invoke({"messages": [("user", "张三 28 岁，爱好爬山")]})
# 结果在 result["structured_response"]
```

## 二、原理讲解

**1. `with_structured_output()` 到底做了什么**

它本质是个 Runnable 包装器，把「模型调用 + 结果解析」打包成一个 `Runnable[输入, BaseModel]`。内部流程是：

```
Pydantic 模型 → model_json_schema() → JSON Schema
    → 绑定为工具（bind_tools）/ 塞进 prompt 的 format_instructions
    → 调模型 → 解析返回值 → Pydantic 校验 → 返回对象
```

**2. 三条实现路径的区别**

| 模式 | 底层机制 | 特点 |
|---|---|---|
| `function_calling` | 把 schema 变成一个 tool，`tool_choice` 强制调用，再从 `tool_calls[0].args` 取参数 | 最稳，主流模型都支持；不支持工具调用的模型用不了 |
| `json_mode` | 设 `response_format={"type":"json_object"}`，同时把格式说明注入 prompt，最后把文本 parse 成 JSON | 兼容性好，但**格式全靠 prompt 保证**，需要自己兜底解析失败 |
| `json_schema` / `strict=True` | 用厂商原生的结构化输出（受约束解码） | 约束最强，几乎不会格式出错，但不是所有模型/网关支持 |

**3. 相比 Parser 方案的核心差别**

`PydanticOutputParser` 是「文本后处理」：模型自由生成 → 抽 JSON → 校验，格式正确率取决于 prompt 质量，失败会抛 `OutputParserException`，要靠 `OutputParserException` + 重试（如 `with_retry`）兜。`with_structured_output` 是「生成时约束」：模型在工具调用/受约束解码阶段就被 schema 限制，基本不用写重试逻辑，所以现在优先用它。

**4. 实践要点**

- `Field(description=...)` 是效果的命脉，字段语义、枚举取值范围、必填与否都要在 description 里写清楚。
- 嵌套模型会展开成 `$defs`，层级别太深，否则部分模型会解析失败。
- 用 `include_raw=True` 观测 `parsing_error`，出问题时先看原始 `args` 是否符合 schema。
- 流式：`with_structured_output` 只在整个对象完成后才吐值；要逐字段流式就用 `JsonOutputParser`。
- 模型兼容性先测：不支持 tool calling 的模型，`function_calling` 会直接报错，退到 `json_mode` 或解析器方案。

需要我再展开哪一块？比如自带的 Pydantic 校验失败后自动重试，或者嵌套模型/枚举的写法。
````







# 5. 总结

1. **Context（运行时上下文）**

   * 通过`runtime.context`访问

   * 用于传递用户ID、配置参数等

   * 生命周期是一次调用

   * 存储方式：基于内存

2. **State（短期记忆）**

   * 通过`runtime.state`参数访问

   * 用于保存会话历史消息、对话状态、计数等临时信息

   * 生命周期是当前会话

   * 存储方式：可以是InMemorySaver、也可以是数据库

3. **Store（长期记忆）**

   * 通过`runtime.store`访问

   * 用于拓展知识、用户偏好等信息

   * 生命周期跨越多个会话

   * 支持InMemoryStore和数据库存储