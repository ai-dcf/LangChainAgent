# 1. 认识Runtime

Runtime是LangChain的Agent运行时对外暴露的对象，其中包含的核心数据有：

| 概念    | 作用                               | 读写   | 典型生命周期                       | 典型内容                               | 访问方式          |
| ------- | ---------------------------------- | ------ | ---------------------------------- | -------------------------------------- | ----------------- |
| Context | 注入本次运行所需的身份、配置和依赖 | 只读   | 单次调用                           | user\_id、tenant\_id、权限、数据库连接 | `runtime.context` |
| State   | 保存 Agent 执行中产生的动态数据    | 可读写 | 同一会话：同一thread\_id的多次调用 | 消息、步骤结果、计数器、业务流程状态   | `runtime.state`   |
| Store   | 保存跨会话需要复用的数据           | 可读写 | 跨会话；                           | 用户偏好、长期记忆、历史资料           | `runtime.store`   |

判断数据放在哪里，可以连续问两个问题：

1. 这份数据在执行过程中会不会变化？不变化，优先考虑 Context。

2. 如果会变化，下一次会话还要不要使用？只在当前会话使用放 State，跨会话复用放 Store。



更多Runtime信息可以查看官方文档：[Runtime - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/runtime)



# 2. Context（运行上下文）

Runtime Context 是一种依赖注入机制：调用者在运行开始时把用户身份、租户信息、权限或外部服务连接传给 Agent，节点、中间件和工具在执行时再从 Runtime 读取。



它解决的是三个工程问题：

* 避免把用户 ID、数据库连接等信息硬编码进工具。

* 同一个 Agent 可以安全地服务不同用户和租户。

* 工具的业务参数只保留真正需要模型填写的内容。



Context 中的数据默认不会自动发给大模型；只有代码主动读取并把内容加入提示词或工具结果时，模型才会看到它。



例如，在系统中用户通常会先登录，然后访问agent，我们就可以把登录用户信息存入Context，在Agent内部的Middleware或Tool旧能很方便的通过`runtime.context`获取用户信息。而这个用户信息并不会让模型看到，也不需要作为参数让模型传递，纯粹是Agent内部信息。



## 2.1 准备数据

下面，我们就以一个多租户的在线教育Agent系统为例，来演示Context的用法。



我们先准备一个字典，模拟一个多租户的在线教育项目数据库，其中的租户就是学校，user就是学员：

```python
# 字典key是租户tenant_id，字典value是租户对应的user信息，也是字典。user字典的key是user_id
USER_DATABASE = {
    "school_a": {
        "u001": {"name": "虎哥", "course": "LangChain 开发指南"},
        "u002": {"name": "小明", "course": "Java开发入门"},
    },
    "school_b": {
        "u001": {"name": "涛哥", "course": "Python 入门"},
        "u002": {"name": "阿伟", "course": "VibeCoding 实战"},
    }
}
```



下面用“查询当前用户资料”演示 Context。模型只负责决定何时调用工具，不负责填写 `user_id` 和 `tenant_id`；工具从可信的 Runtime Context 中读取它们。



## 2.2 定义Context Schema

首先，我们需要约定Context的数据结构，这里要使用`@dataclass`来定义，省去定义`__init__`等魔法函数



示例代码：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class UserContext:
    """Agent运行时上下文"""
    user_id: str
    tenant_id: str

```



说明：

> 定义UserContext类就是一个`context_schema` ，作用是给 Context 提供明确的数据结构。`frozen=True` 不是 LangChain 的硬性要求，但能表达“运行内不可修改”的设计意图。



## 2.3 在Tool中使用Context

在tool中，我们可以添加runtime参数，然后利用`runtime.context`来访问Context：

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_current_user_profile(runtime: ToolRuntime[UserContext]):
    """查询当前登录用户的资料。"""

    # 获取context
    ctx= runtime.context

    # 根据user_id和tenant_id获取用户信息
    profile = USER_DATABASE.get(ctx.tenant_id, {}).get(ctx.user_id)
    
    return (
        f"姓名：{profile['name']}，"
        f"课程：{profile['course']}，"
    )
```



## 2.4 给Agent添加Context

在定义Agent时，需要指定我们创建好的Context：

```python
from langchain.agents import create_agent

# 创建Agent时指定context_schema
agent = create_agent(
    model="deepseek-flash",
    tools=[get_current_user_profile],
    context_schema=UserContext,  # 指定Context类型
    system_prompt="你是课程平台助手。需要用户资料时调用工具。"
)
```

调用Agent时，可以传递Context信息：

```python
from langchain.messages import HumanMessage

# 调用时通过Context传递用户信息
response = agent.invoke(
    {"messages": [HumanMessage("Hello, 帮我查询我正在学习的课程")]},
    context=UserContext(user_id="u001", tenant_id="school_a")
)

for message in response['messages']:
    message.pretty_print()
```
